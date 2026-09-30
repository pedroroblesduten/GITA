from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron with optional LayerNorm after each activation."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LengthNormalize(nn.Module):
    """Normalize the last dimension to length sqrt(dim)."""

    @nn.compact
    def __call__(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-8) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and variance for normalization (clipped to clip_max)."""

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned Gaussian actor."""

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution; temperature scales the std."""
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions."""

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution; logits are divided by temperature (0 gives argmax)."""
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class CMGCActor(nn.Module):
    """Goal-conditioned actor also conditioned on log k (FiLM if use_film, else concatenation)."""

    hidden_dims: Sequence[int]
    action_dim: int
    abstraction_hidden_dims: Sequence[int] = (32,)
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    use_film: bool = False

    def setup(self):
        if self.use_film:
            self.actor_net = FiLMMLP(self.hidden_dims, activate_final=True)
        else:
            self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.abstraction_net = MLP(self.abstraction_hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def _abstraction_emb(self, observations, log_abstraction_factor):
        """Embed log k (scalar or per-example) into a per-example feature."""
        batch_shape = observations.shape[:-1]
        log_af = jnp.asarray(log_abstraction_factor, dtype=observations.dtype)
        if log_af.ndim == 0:
            log_af = jnp.broadcast_to(log_af, batch_shape)
        log_af = log_af.reshape(batch_shape + (1,))
        return self.abstraction_net(log_af)

    def __call__(
        self,
        observations,
        goals=None,
        log_abstraction_factor=0.0,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution at log k (default 0.0, i.e. k = 1); temperature scales the std."""
        if self.gc_encoder is not None:
            representation = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            representation = jnp.concatenate(inputs, axis=-1)

        abstraction_emb = self._abstraction_emb(observations, log_abstraction_factor)

        if self.use_film:
            outputs = self.actor_net(representation, abstraction_emb)
        else:
            inputs = jnp.concatenate([representation, abstraction_emb], axis=-1)
            outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value V(s, g) or critic Q(s, a, g); agents assume num_qs=2 heads."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    num_qs: int = 2
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, self.num_qs)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function."""
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v

    def value_apply_encoded_goal(self, observations, goal_reps, actions=None):
        """Return the value/critic function given already-encoded goals (e.g. goal_rep outputs)."""
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goal_reps, goal_encoded=True)]
        else:
            inputs = [observations, goal_reps]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class FiLMMLP(nn.Module):
    """MLP whose activated layers are FiLM-modulated, x <- (1 + gamma) * x + beta, after LayerNorm.

    FiLM weights start at zero, so the network starts as a plain MLP.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, condition):
        # x: (..., input_dim)
        # condition: (..., condition_dim)

        num_layers = len(self.hidden_dims)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(
                size,
                kernel_init=self.kernel_init,
                name=f'dense_{i}',
            )(x)  # (..., size)

            if i + 1 < num_layers or self.activate_final:
                x = self.activations(x)  # (..., size)

                if self.layer_norm:
                    x = nn.LayerNorm(
                        name=f'layer_norm_{i}',
                    )(x)  # (..., size)

                film_params = nn.Dense(
                    2 * size,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                    name=f'film_{i}',
                )(condition)  # (..., 2 * size)

                gamma, beta = jnp.split(
                    film_params,
                    2,
                    axis=-1,
                )  # each: (..., size)

                x = (1.0 + gamma) * x + beta  # (..., size)

        return x  # (..., output_dim)


class CMGCValue(nn.Module):
    """Goal-conditioned value V(s, g; k), conditioned on log k via FiLM (use_film) or concatenation.

    The state/goal encoder never sees k. Agents assume num_qs=2 heads.
    """

    hidden_dims: Sequence[int]
    abstraction_hidden_dims: Sequence[int] = (32,)
    layer_norm: bool = True
    ensemble: bool = True
    num_qs: int = 2
    gc_encoder: nn.Module = None
    use_film: bool = False

    def setup(self):
        if self.use_film:
            value_module = FiLMMLP

            if self.ensemble:
                value_module = ensemblize(value_module, self.num_qs)

            self.value_net = value_module(
                hidden_dims=(*self.hidden_dims, 1),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

        else:
            value_module = MLP

            if self.ensemble:
                value_module = ensemblize(value_module, self.num_qs)

            self.value_net = value_module(
                (*self.hidden_dims, 1),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

        # log k embedding; no LayerNorm, to keep its magnitude.
        self.abstraction_net = MLP(
            self.abstraction_hidden_dims,
            activate_final=True,
        )

    def _abstraction_emb(self, batch_shape, dtype, log_abstraction_factor):
        """Embed log k; batch_shape/dtype come from the encoded representation, not raw (image) observations."""

        log_af = jnp.asarray(
            log_abstraction_factor,
            dtype=dtype,
        )  # batch_shape or scalar

        if log_af.ndim == 0:
            log_af = jnp.broadcast_to(
                log_af,
                batch_shape,
            )  # batch_shape

        log_af = log_af.reshape(
            batch_shape + (1,)
        )  # (*batch_shape, 1)

        abstraction_emb = self.abstraction_net(
            log_af
        )  # (*batch_shape, abstraction_hidden_dims[-1])

        return abstraction_emb

    def encode(self, observations, goals=None, actions=None):
        """Encode (observations, goals[, actions]) once, independent of k, for reuse across factors."""

        if self.gc_encoder is not None:
            representation = self.gc_encoder(
                observations,
                goals,
            )  # (*batch_shape, representation_dim)

        else:
            inputs = [observations]

            if goals is not None:
                inputs.append(goals)

            representation = jnp.concatenate(
                inputs,
                axis=-1,
            )  # (*batch_shape, representation_dim)

        if actions is not None:
            representation = jnp.concatenate(
                [representation, actions],
                axis=-1,
            )  # (*batch_shape, representation_dim + action_dim)

        return representation

    def value_from_representation(self, representation, log_abstraction_factor=0.0):
        """Value head only, given an already-encoded representation (see encode)."""

        abstraction_emb = self._abstraction_emb(
            representation.shape[:-1],
            representation.dtype,
            log_abstraction_factor,
        )  # (*batch_shape, abstraction_dim)

        if self.use_film:
            v = self.value_net(
                representation,
                abstraction_emb,
            ).squeeze(-1)  # (num_heads, *batch_shape) if ensemble

        else:
            representation = jnp.concatenate(
                [representation, abstraction_emb],
                axis=-1,
            )  # (*batch_shape, representation_dim + abstraction_dim)

            v = self.value_net(
                representation
            ).squeeze(-1)  # (num_heads, *batch_shape) if ensemble

        return v

    def __call__(
        self,
        observations,
        goals=None,
        log_abstraction_factor=0.0,
        actions=None,
    ):
        """Return the value/critic prediction."""
        representation = self.encode(observations, goals, actions)
        return self.value_from_representation(representation, log_abstraction_factor)

    def value_apply_encoded_goal(
        self,
        observations,
        goal_reps,
        log_abstraction_factor=0.0,
        actions=None,
    ):
        """Return value given an already-encoded goal representation."""

        if self.gc_encoder is not None:
            representation = self.gc_encoder(
                observations,
                goal_reps,
                goal_encoded=True,
            )  # (*batch_shape, representation_dim)

        else:
            representation = jnp.concatenate(
                [observations, goal_reps],
                axis=-1,
            )  # (*batch_shape, representation_dim)

        if actions is not None:
            representation = jnp.concatenate(
                [representation, actions],
                axis=-1,
            )  # (*batch_shape, representation_dim + action_dim)

        return self.value_from_representation(representation, log_abstraction_factor)


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Bilinear value/critic: V(s, g) = phi(s)^T psi(g) / sqrt(d), or phi(s, a) for Q."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function (and phi, psi if info)."""
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """MRN value: symmetric Euclidean distance plus an asymmetric L-infinity quasimetric."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value (and phi_s, phi_g if info); is_phi means inputs are already encoded."""
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function."""

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value (and phi_s, phi_g if info); is_phi means inputs are already encoded."""
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v
