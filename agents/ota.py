from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize


class OTAAgent(flax.struct.PyTreeNode):
    """Option-aware temporally abstracted Value Learning (OTA) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def low_value_loss(self, batch, grad_params):
        """Compute the IVL value loss."""
        (next_v1_t, next_v2_t) = self.network.select('target_low_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['low_discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_low_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['low_discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['low_discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('low_value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        low_value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        low_value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        low_value_loss = low_value_loss1 + low_value_loss2

        return low_value_loss, {
            'low_value_loss': low_value_loss,
            'low_v_mean': v.mean(),
            'low_v_max': v.max(),
            'low_v_min': v.min(),
        }

    def high_value_loss(self, batch, grad_params):
        """Compute the option-aware value loss."""
        high_value_option_observations = batch['high_value_option_observations']
        high_value_goals = batch['high_value_goals']
        high_value_rewards = batch['high_value_rewards']
        high_value_masks = batch['high_value_masks']

        (next_v1_t, next_v2_t) = self.network.select('target_high_value')(high_value_option_observations, high_value_goals)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = high_value_rewards + self.config['high_discount'] * high_value_masks * next_v_t

        (v1_t, v2_t) = self.network.select('target_high_value')(batch['observations'], high_value_goals)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = high_value_rewards + self.config['high_discount'] * high_value_masks * next_v1_t
        q2 = high_value_rewards + self.config['high_discount'] * high_value_masks * next_v2_t
        (v1, v2) = self.network.select('high_value')(batch['observations'], high_value_goals, params=grad_params)
        v = (v1 + v2) / 2

        high_value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        high_value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        high_value_loss = high_value_loss1 + high_value_loss2
        
        return high_value_loss, {
            'high_value_loss': high_value_loss,
            'high_v_mean': v.mean(),
            'high_v_max': v.max(),
            'high_v_min': v.min()
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss.

        Also returns (current_state, encoded_target) for coord_decoder_loss, avoiding a redundant goal_rep forward pass.
        """
        v1, v2 = self.network.select('low_value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('low_value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        target_reps = {
            'current_state': batch['observations'],
            'encoded_target': goal_reps,
        }

        return actor_loss, actor_info, target_reps

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss (horizon = subgoal_steps).

        Also returns (current_state, encoded_target) — the AWR target — for coord_decoder_loss.
        """
        high_actor_goals = batch['high_actor_goals']
        high_actor_targets = batch['high_actor_targets']

        v1, v2 = self.network.select('high_value')(batch['observations'], high_actor_goals)
        nv1, nv2 = self.network.select('high_value')(high_actor_targets, high_actor_goals)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], high_actor_goals, params=grad_params)
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], high_actor_targets], axis=-1)
        )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }
        target_reps = {
            'current_state': batch['observations'],
            'encoded_target': target,
        }

        return actor_loss, actor_info, target_reps

    def coord_decoder_loss(self, batch, grad_params, low_target_reps, high_target_reps):
        """MSE loss for the low- and high-level coordinate decoders: predict (x, y) from current state + encoded target.

        Both decoders are independent (inputs stop-gradiented here), so this loss can't affect any other network.
        """
        info = {}

        low_current_state = jax.lax.stop_gradient(low_target_reps['current_state'])
        low_encoded_target = jax.lax.stop_gradient(low_target_reps['encoded_target'])
        low_coords = batch['low_actor_goals'][:, :2]
        low_inputs = jnp.concatenate([low_current_state, low_encoded_target], axis=-1)
        low_pred_coords = self.network.select('coord_decoder_low')(low_inputs, params=grad_params)
        low_loss = jnp.mean((low_pred_coords - low_coords) ** 2)
        info['coord_decoder_loss_low'] = low_loss

        high_current_state = jax.lax.stop_gradient(high_target_reps['current_state'])
        high_encoded_target = jax.lax.stop_gradient(high_target_reps['encoded_target'])
        high_coords = batch['high_actor_targets'][:, :2]
        high_inputs = jnp.concatenate([high_current_state, high_encoded_target], axis=-1)
        high_pred_coords = self.network.select('coord_decoder_high')(high_inputs, params=grad_params)
        high_loss = jnp.mean((high_pred_coords - high_coords) ** 2)
        info['coord_decoder_loss_high'] = high_loss

        total_loss = low_loss + high_loss
        info['coord_decoder_loss'] = total_loss
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f'low_value/{k}'] = v

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        low_actor_loss, low_actor_info, low_target_reps = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info, high_target_reps = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        coord_decoder_loss, coord_decoder_info = self.coord_decoder_loss(
            batch, grad_params, low_target_reps, high_target_reps
        )
        for k, v in coord_decoder_info.items():
            info[f'coord_decoder/{k}'] = v

        loss = low_value_loss + high_value_loss + low_actor_loss + high_actor_loss + coord_decoder_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, 'low_value')
        self.target_update(new_network, 'high_value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions: high_actor picks a subgoal rep, then low_actor conditions on it for raw actions.

        Also returns goal_reps and its decoded (x, y) coord (via coord_decoder_high), each wrapped in a
        length-1 list for interface parity with multi-stage agents. Note: goal_reps matches the
        high_actor_targets encoding, not low_actor_goals.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        coord_input = jnp.concatenate([observations, goal_reps], axis=-1)
        subgoal_coord = self.network.select('coord_decoder_high')(coord_input)

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions, [goal_reps], [subgoal_coord]

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Subgoal representation phi([s; g]), length-normalized.
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)         

        if config['encoder'] is not None:
            low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_high_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_high_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None
        
        # Define value and actor networks.
        low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=low_value_encoder_def,
        )
        target_low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_low_value_encoder_def,
        )

        high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=high_value_encoder_def,
        )
        target_high_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_high_value_encoder_def,
        )
        
        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        # Coordinate decoders: independent MLPs predicting (x, y) from current state + encoded target.
        # One per stage (low, high); not referenced by any other loss.
        coord_decoder_low_def = MLP(
            hidden_dims=(*config['value_hidden_dims'], 2),
            activate_final=False,
            layer_norm=config['layer_norm'],
        )
        coord_decoder_high_def = MLP(
            hidden_dims=(*config['value_hidden_dims'], 2),
            activate_final=False,
            layer_norm=config['layer_norm'],
        )
        ex_coord_input = jnp.concatenate(
            [
                ex_observations.reshape((ex_observations.shape[0], -1)),
                jnp.zeros((ex_observations.shape[0], config['rep_dim'])),
            ],
            axis=-1,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            low_value=(low_value_def, (ex_observations, ex_goals)),
            target_low_value=(target_low_value_def, (ex_observations, ex_goals)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            target_high_value=(target_high_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            coord_decoder_low=(coord_decoder_low_def, (ex_coord_input,)),
            coord_decoder_high=(coord_decoder_high_def, (ex_coord_input,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_low_value'] = params['modules_low_value']
        params['modules_target_high_value'] = params['modules_high_value']

        config['low_discount'] = config['discount']
        config['high_discount'] = config['discount']

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


