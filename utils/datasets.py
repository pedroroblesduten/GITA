import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an edge-padded image."""
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Frozen dict of arrays; next_observations is inferred from observations when absent."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields, optionally freezing the arrays."""
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Dataset that supports adding transitions."""

    @classmethod
    def create(cls, transition, size):
        """Create an empty buffer shaped like the example transition."""

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a buffer pre-filled with init_dataset."""

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned dataset: goals are the current state, a future state in the trajectory, or a random state."""

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample transitions with value/actor goals, rewards, and masks (no augmentation when evaluation=True)."""
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)

        def crop(arr, froms):
            return np.array(batched_random_crop(arr, froms, padding)) if len(arr.shape) == 4 else arr

        for key in keys:
            value = batch[key]
            batch[key] = jax.tree_util.tree_map(lambda arr: crop(arr, crop_froms), value)

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Hierarchical dataset: adds actor goals/subgoal targets, plus high_value_* (ota) or geo_* (gita) value targets."""

    def sample_high_goals(
        self,
        idxs,
        p_curgoal,
        p_trajgoal,
        p_randomgoal,
        geom_sample,
    ):
        """Sample OTA's (goal, option-terminal) indices with the fixed option length abstraction_factor."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(
            batch_size
        )

        # Goals from the same trajectory.
        final_state_idxs = self.terminal_locs[
            np.searchsorted(
                self.terminal_locs,
                idxs,
            )
        ]

        if geom_sample:
            offsets = np.random.geometric(
                p=1 - self.config['discount'],
                size=batch_size,
            )

            traj_goal_idxs = np.minimum(
                idxs + offsets,
                final_state_idxs,
            )
        else:
            distances = np.random.rand(
                batch_size
            )

            traj_goal_idxs = np.round(
                (
                    np.minimum(
                        idxs + 1,
                        final_state_idxs,
                    )
                    * distances
                    + final_state_idxs
                    * (1 - distances)
                )
            ).astype(int)

        # Whether to sample from the same trajectory.
        traj_masks = (
            np.random.rand(batch_size)
            < p_trajgoal / (1.0 - p_curgoal + 1e-6)
        )

        goal_idxs = np.where(
            traj_masks,
            traj_goal_idxs,
            random_goal_idxs,
        )

        cur_masks = (
            np.random.rand(batch_size)
            < p_curgoal
        )

        abstraction_factor = self.config['abstraction_factor']

        # Timeout-based option termination.
        original_subgoal_idxs = np.minimum(
            idxs + abstraction_factor,
            final_state_idxs,
        )

        # Goal-based early termination.
        option_subgoal_idxs = np.minimum(
            idxs + abstraction_factor,
            traj_goal_idxs,
        )

        high_value_option_idxs = np.where(
            traj_masks,
            option_subgoal_idxs,
            original_subgoal_idxs,
        )

        high_value_goal_idxs = np.where(
            cur_masks,
            high_value_option_idxs,
            goal_idxs,
        )

        return (
            high_value_goal_idxs,
            high_value_option_idxs,
        )

    def sample_geo_value_goal(
        self,
        idxs,
        p_curgoal,
        p_trajgoal,
        p_randomgoal,
        geom_sample,
    ):
        """Sample GITA's (goal, option-terminal, timeout) indices; the timeout is censored by goal or trajectory end."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(
            batch_size
        )

        # Goals from the same trajectory.
        final_state_idxs = self.terminal_locs[
            np.searchsorted(
                self.terminal_locs,
                idxs,
            )
        ]

        if geom_sample:
            offsets = np.random.geometric(
                p=1 - self.config['discount'],
                size=batch_size,
            )

            traj_goal_idxs = np.minimum(
                idxs + offsets,
                final_state_idxs,
            )
        else:
            distances = np.random.rand(
                batch_size
            )

            traj_goal_idxs = np.round(
                (
                    np.minimum(
                        idxs + 1,
                        final_state_idxs,
                    )
                    * distances
                    + final_state_idxs
                    * (1 - distances)
                )
            ).astype(int)

        traj_masks = (
            np.random.rand(batch_size)
            < p_trajgoal / (1.0 - p_curgoal + 1e-6)
        )

        goal_idxs = np.where(
            traj_masks,
            traj_goal_idxs,
            random_goal_idxs,
        )

        cur_masks = (
            np.random.rand(batch_size)
            < p_curgoal
        )

        # Option timeout k ~ Geom(1 - geo_gamma); old configs without geo_gamma use discount.
        geo_option_timeouts = np.random.geometric(
            p=1 - self.config.get('geo_gamma', self.config['discount']),
            size=batch_size,
        )

        timeout_terminal_idxs = np.minimum(
            idxs + geo_option_timeouts,
            final_state_idxs,
        )

        goal_or_timeout_terminal_idxs = np.minimum(
            idxs + geo_option_timeouts,
            traj_goal_idxs,
        )

        geo_option_terminal_idxs = np.where(
            traj_masks,
            goal_or_timeout_terminal_idxs,
            timeout_terminal_idxs,
        )

        geo_value_goal_idxs = np.where(
            cur_masks,
            geo_option_terminal_idxs,
            goal_idxs,
        )

        return (
            geo_value_goal_idxs,
            geo_option_terminal_idxs,
            geo_option_timeouts,
        )

    def sample(
        self,
        batch_size,
        idxs=None,
        evaluation=False,
    ):
        """Sample a batch of hierarchical goal-conditioned transitions."""
        if idxs is None:
            idxs = self.dataset.get_random_idxs(
                batch_size
            )

        idxs = np.asarray(
            idxs,
            dtype=np.int64,
        )

        batch_size = len(idxs)  # Actual size when idxs is given.

        batch = self.dataset.sample(
            batch_size,
            idxs,
        )
        batch['idxs'] = idxs

        if self.config['frame_stack'] is not None:
            batch['observations'] = (
                self.get_observations(
                    idxs
                )
            )

            batch['next_observations'] = (
                self.get_observations(
                    idxs + 1
                )
            )

        # OTA option-aware value targets.
        if self.config['agent_name'] == 'ota':
            (
                high_value_goal_idxs,
                high_value_option_idxs,
            ) = self.sample_high_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                self.config['value_geom_sample'],
            )

            batch['high_value_goals'] = (
                self.get_observations(
                    high_value_goal_idxs
                )
            )

            batch['high_value_option_observations'] = (
                self.get_observations(
                    high_value_option_idxs
                )
            )

            high_value_success = (
                high_value_option_idxs
                == high_value_goal_idxs
            ).astype(float)

            batch['high_value_masks'] = (
                1.0 - high_value_success
            )

            batch['high_value_rewards'] = (
                high_value_success
                - (
                    1.0
                    if self.config['gc_negative']
                    else 0.0
                )
            )

        # GITA value targets.
        if self.config['agent_name'] == 'gita':
            (
                geo_value_goal_idxs,
                geo_option_terminal_idxs,
                geo_option_timeouts,
            ) = self.sample_geo_value_goal(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                self.config['value_geom_sample'],
            )

            batch['geo_option_timeouts'] = (
                geo_option_timeouts
            )

            batch['geo_option_durations'] = (
                geo_option_terminal_idxs - idxs
            )

            batch['geo_option_observations'] = (
                self.get_observations(
                    geo_option_terminal_idxs
                )
            )

            batch['geo_value_goals'] = (
                self.get_observations(
                    geo_value_goal_idxs
                )
            )

            geo_success = (
                geo_option_terminal_idxs
                == geo_value_goal_idxs
            ).astype(float)

            batch['geo_masks'] = (
                1.0 - geo_success
            )

            batch['geo_rewards'] = (
                geo_success
                - (
                    1.0
                    if self.config['gc_negative']
                    else 0.0
                )
            )

        # One-step value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )

        batch['value_goals'] = (
            self.get_observations(
                value_goal_idxs
            )
        )

        successes = (
            idxs == value_goal_idxs
        ).astype(float)

        batch['masks'] = (
            1.0 - successes
        )

        batch['rewards'] = (
            successes
            - (
                1.0
                if self.config['gc_negative']
                else 0.0
            )
        )

        # Low-level actor goals.
        final_state_idxs = self.terminal_locs[
            np.searchsorted(
                self.terminal_locs,
                idxs,
            )
        ]

        low_goal_idxs = np.minimum(
            idxs + self.config['subgoal_steps'],
            final_state_idxs,
        )

        batch['low_actor_goals'] = (
            self.get_observations(
                low_goal_idxs
            )
        )

        # High-level actor goals and subgoal targets.
        if self.config['actor_geom_sample']:
            offsets = np.random.geometric(
                p=1 - self.config['discount'],
                size=batch_size,
            )

            high_traj_goal_idxs = np.minimum(
                idxs + offsets,
                final_state_idxs,
            )
        else:
            distances = np.random.rand(
                batch_size
            )

            high_traj_goal_idxs = np.round(
                (
                    np.minimum(
                        idxs + 1,
                        final_state_idxs,
                    )
                    * distances
                    + final_state_idxs
                    * (1 - distances)
                )
            ).astype(int)

        high_random_goal_idxs = (
            self.dataset.get_random_idxs(
                batch_size
            )
        )

        pick_random = (
            np.random.rand(batch_size)
            < self.config['actor_p_randomgoal']
        )

        high_goal_idxs = np.where(
            pick_random,
            high_random_goal_idxs,
            high_traj_goal_idxs,
        )

        high_traj_target_idxs = np.minimum(
            idxs + self.config['subgoal_steps'],
            high_traj_goal_idxs,
        )

        high_random_target_idxs = np.minimum(
            idxs + self.config['subgoal_steps'],
            final_state_idxs,
        )

        high_target_idxs = np.where(
            pick_random,
            high_random_target_idxs,
            high_traj_target_idxs,
        )

        batch['high_actor_goals'] = (
            self.get_observations(
                high_goal_idxs
            )
        )

        batch['high_actor_targets'] = (
            self.get_observations(
                high_target_idxs
            )
        )

        batch['high_actor_target_idxs'] = (
            high_target_idxs
        )

        # Image augmentation.
        if (
            self.config['p_aug'] is not None
            and not evaluation
        ):
            if (
                np.random.rand()
                < self.config['p_aug']
            ):
                augment_keys = [
                    'observations',
                    'next_observations',
                    'value_goals',
                    'low_actor_goals',
                    'high_actor_goals',
                    'high_actor_targets',
                ]

                self.augment(
                    batch,
                    augment_keys,
                )

        return batch
