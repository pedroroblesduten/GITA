# gita.py

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, CMGCValue, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize

class GITAAgent(flax.struct.PyTreeNode):
    """Generalized Implicit Temporal Abstraction (GITA) agent"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def encode_state_and_goal(self, observations, goals):
        """Encode (observations, goals) once via value's encoder, no gradient."""
        return self.network.select('value')(
            observations, goals, submodule_method='encode',
        )

    def value_loss(self, batch, grad_params):
        """IVL loss conditioned on realized duration m: q = r + gamma * mask * V_target(s^Omega, g, m), loss = expectile(q - V(s, g, m))."""
        option_durations = jnp.maximum(batch['geo_option_durations'], 1)
        log_option_durations = jnp.log(option_durations.astype(jnp.float32))

        (next_v1_t, next_v2_t) = self.network.select('target_value')(
            batch['geo_option_observations'], batch['geo_value_goals'], log_option_durations
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['geo_rewards'] + self.config['discount'] * batch['geo_masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(
            batch['observations'], batch['geo_value_goals'], log_option_durations
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['geo_rewards'] + self.config['discount'] * batch['geo_masks'] * next_v1_t
        q2 = batch['geo_rewards'] + self.config['discount'] * batch['geo_masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(
            batch['observations'], batch['geo_value_goals'], log_option_durations, params=grad_params
        )

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {'value_loss': value_loss}

    def low_value_loss(self, batch, grad_params):
        """One-step IVL loss for low_value: q = r + gamma * mask * V_target(s', g), loss = expectile(q - V(s, g))."""
        (next_v1_t, next_v2_t) = self.network.select('target_low_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_low_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('low_value')(batch['observations'], batch['value_goals'], params=grad_params)

        low_value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        low_value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        low_value_loss = low_value_loss1 + low_value_loss2

        return low_value_loss, {'low_value_loss': low_value_loss}

    def low_actor_loss(self, batch, grad_params):
        """AWR low-actor loss: weight = min(exp(low_alpha * (V(s', g) - V(s, g))), 100)."""
        v1, v2 = self.network.select('low_value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('low_value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {'actor_loss': actor_loss}

        return actor_loss, actor_info

    @staticmethod
    def _tile_over_factors(values, num_factors):
        """Repeat a batch over factors and flatten the first two axes."""
        batch_size = values.shape[0]
        return jnp.broadcast_to(
            values[:, None, ...],
            (batch_size, num_factors, *values.shape[1:]),
        ).reshape((batch_size * num_factors, *values.shape[1:]))

    def high_actor_loss(self, batch, grad_params):
        """Train one unconditioned actor with uniform post-exp aggregation.

        W = mean_k min(exp(high_alpha * A_k), 100), where the mean is over
        finite per-factor advantages. Samples with no finite advantage do not
        contribute to the loss. Values receive no gradients from this loss.
        """
        high_actor_targets = batch['high_actor_targets']  # (B, *obs_shape)
        high_actor_goals = batch['high_actor_goals']  # (B, *goal_shape)
        observations = batch['observations']
        batch_size = observations.shape[0]
        factors = jnp.asarray(
            self.config['abstraction_factors'], dtype=observations.dtype
        )
        num_factors = factors.shape[0]

        tiled_factors = jnp.broadcast_to(
            factors[None, :], (batch_size, num_factors)
        ).reshape((batch_size * num_factors,))
        tiled_log_factors = jnp.log(tiled_factors)

        state_target = jnp.concatenate([observations, high_actor_targets], axis=0)  # (2B, *obs_shape)
        goal_pair = jnp.concatenate([high_actor_goals, high_actor_goals], axis=0)  # (2B, *goal_shape)
        representation = self.encode_state_and_goal(state_target, goal_pair)  # (2B, rep_dim)
        tiled_representation = self._tile_over_factors(representation, num_factors)  # (2B*K, rep_dim)
        log_factor_pair = jnp.concatenate([tiled_log_factors, tiled_log_factors], axis=0)  # (2B*K,)

        value_heads = self.network.select('value')(
            tiled_representation, log_factor_pair, submodule_method='value_from_representation',
        )  # (2, 2*B*K)

        half = batch_size * num_factors
        state_values = jnp.mean(value_heads[:, :half], axis=0)  # (B*K,)
        target_values = jnp.mean(value_heads[:, half:], axis=0)  # (B*K,)
        raw_adv = (target_values - state_values).reshape((batch_size, num_factors))  # (B, K)
        raw_adv = jax.lax.stop_gradient(raw_adv)
        advantage_valid = jnp.isfinite(raw_adv)
        adv = jnp.where(advantage_valid, raw_adv, 0.0)

        exp_a = jnp.minimum(
            jnp.exp(adv * self.config['high_alpha']),
            100.0,
        )
        exp_a = jax.lax.stop_gradient(exp_a)

        valid_float = advantage_valid.astype(adv.dtype)
        factor_count = valid_float.sum(axis=-1)
        sample_valid = factor_count > 0
        actor_weight = jax.lax.stop_gradient(
            jnp.where(
                sample_valid,
                (exp_a * valid_float).sum(axis=-1)
                / jnp.maximum(factor_count, 1.0),
                0.0,
            )
        )

        dist = self.network.select('high_actor')(
            observations,
            high_actor_goals,
            params=grad_params,
        )

        encoded_target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], high_actor_targets], axis=-1)
        )  # (B, rep_dim)
        log_prob = dist.log_prob(encoded_target)

        sample_valid_float = sample_valid.astype(log_prob.dtype)
        sample_denom = jnp.maximum(sample_valid_float.sum(), 1.0)
        actor_loss = -(actor_weight * log_prob * sample_valid_float).sum() / sample_denom

        advantage_denom = jnp.maximum(valid_float.sum(), 1.0)
        actor_info = {
            'actor_loss': actor_loss,
            'raw_adv_mean': (adv * valid_float).sum() / advantage_denom,
            'awr_weight_mean': (actor_weight * sample_valid_float).sum() / sample_denom,
            'valid_fraction': valid_float.mean(),
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Sum of value_loss, low_value_loss, low_actor_loss, and high_actor_loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f'low_value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_value_loss + low_actor_loss + high_actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Polyak-average the target network: target = tau * p + (1 - tau) * target."""
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

        self.target_update(new_network, 'value')
        self.target_update(new_network, 'low_value')

        return self.replace(
            network=new_network,
            rng=new_rng,
        ), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample an unconditioned subgoal and a low-level action, without factor selection."""
        high_seed, low_seed = jax.random.split(seed)

        dist = self.network.select('high_actor')(
            observations,
            goals,
            temperature=temperature,
        )
        subgoal_rep = dist.sample(seed=high_seed)
        subgoal_rep = (
            subgoal_rep
            / (jnp.linalg.norm(subgoal_rep, axis=-1, keepdims=True) + 1e-8)
            * jnp.sqrt(subgoal_rep.shape[-1])
        )

        low_dist = self.network.select('low_actor')(
            observations, subgoal_rep, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        return actions, [subgoal_rep], [None]

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Build the network (value/low_value/high_actor/low_actor/goal_rep, plus targets) and return a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

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
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None

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
        value_def = CMGCValue(
            hidden_dims=config['value_hidden_dims'],
            abstraction_hidden_dims=config['abstraction_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            num_qs=2,
            gc_encoder=value_encoder_def,
            use_film=config['use_film'],
        )
        target_value_def = CMGCValue(
            hidden_dims=config['value_hidden_dims'],
            abstraction_hidden_dims=config['abstraction_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            num_qs=2,
            gc_encoder=target_value_encoder_def,
            use_film=config['use_film'],
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

        goal_rep_input = jnp.concatenate([ex_observations, ex_goals], axis=-1)
        network_info = dict(
            goal_rep=(goal_rep_def, (goal_rep_input,)),
            low_value=(low_value_def, (ex_observations, ex_goals)),
            target_low_value=(target_low_value_def, (ex_observations, ex_goals)),
            value=(value_def, (ex_observations, ex_goals, 0.0)),
            target_value=(target_value_def, (ex_observations, ex_goals, 0.0)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_low_value'] = params['modules_low_value']

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )
