from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cfp._types import ArrayLike
from cfp.networks._velocity_field import ConditionalVelocityField

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        flow
            Flow between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`cfp.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure flow matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        cfg_p_resample
            Probability of the null condition for classifier free guidance.
        cfg_ode_weight
            Weighting factor of the null condition for classifier free guidance.
            0 corresponds to no classifier-free guidance, the larger 0, the more guidance.
        kwargs
            Keyword arguments for :meth:`cfp.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        flow: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[
            [jax.Array, int], jnp.ndarray
        ] = solver_utils.uniform_sampler,
        cfg_p_resample: float = 0.0,
        cfg_ode_weight: float = 0.0,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.flow = flow
        self.time_sampler = time_sampler
        if cfg_p_resample > 0 and cfg_ode_weight == 0:
            raise ValueError(
                "cfg_p_resample > 0 requires cfg_ode_weight > 0 for classifier free guidance."
            )
        if cfg_p_resample == 0 and cfg_ode_weight > 0:
            raise ValueError(
                "cfg_ode_weight > 0 requires cfg_p_resample > 0 for classifier free guidance."
            )
        if cfg_ode_weight < 0:
            raise ValueError("cfg_ode_weight must be non-negative.")
        self.cfg_p_resample = cfg_p_resample
        self.cfg_ode_weight = cfg_ode_weight
        self.match_fn = jax.jit(match_fn)

        self.vf_state = self.vf.create_train_state(
            input_dim=self.vf.output_dims[-1], **kwargs
        )
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]

        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray] | None,
        ) -> tuple[Any, Any]:

            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray] | None,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_dropout = jax.random.split(rng, 2)
                x_t = self.flow.compute_xt(rng_flow, t, source, target)
                v_t = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    rngs={"dropout": rng_dropout},
                )
                u_t = self.flow.compute_ut(t, source, target)

                return jnp.mean((v_t - u_t) ** 2)

            batch_size = len(source)
            key_t, key_model = jax.random.split(rng, 2)
            t = self.time_sampler(key_t, batch_size)
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(
                vf_state.params, t, source, target, conditions, key_model
            )
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys `src_cell_data`, `tgt_cell_data`, and
            optionally `condition`.

        Returns
        -------
        Loss value.
        """
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_cfg, rng_step_fn = jax.random.split(rng, 3)
        cfg_null = jax.random.bernoulli(rng_cfg, self.cfg_p_resample)
        if cfg_null:
            # TODO: adapt to null condition in transformer
            condition = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), condition)

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            src,
            tgt,
            condition,
        )
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike]) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode

        Returns
        -------
        Encoded conditions.
        """
        cond_embed = self.vf.apply(
            {"params": self.vf_state.params},
            condition,
            method="get_condition_embedding",
        )
        return np.asarray(cond_embed)

    def predict(
        self, x: ArrayLike, condition: dict[str, ArrayLike], **kwargs: Any
    ) -> ArrayLike:
        """Predict the translated source ``'x'`` under condition ``'condition'``.

        This function solves the ODE learnt with
        the :class:`~cfp.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            Input data of shape [batch_size, ...].
        condition
            Condition of the input data of shape [batch_size, ...].
        kwargs
            Keyword arguments for :func:`~diffrax.odesolve`.

        Returns
        -------
        The push-forward distribution of ``'x'`` under condition ``'condition'``.
        """
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault(
            "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
        )
        null_cond = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), condition)
            

        def vf(
            t: jnp.ndarray, x: jnp.ndarray, cond: dict[str, jnp.ndarray] | None
        ) -> jnp.ndarray:
            params = self.vf_state.params
            return self.vf_state.apply_fn({"params": params}, t, x, cond, train=False)

        def vf_cfg(
            t: jnp.ndarray, x: jnp.ndarray, cond: dict[str, jnp.ndarray] | None
        ) -> jnp.ndarray:
            cond = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), cond)
            params = self.vf_state.params
            return (1 + self.cfg_ode_weight) * self.vf_state.apply_fn({"params": params}, t, x, cond, train=False)

            # TODO: adapt to null condition in transformer
            #params = self.vf_state.params
            #v_cond = self.vf_state.apply_fn({"params": params}, t, x, cond)
            #v_uncond = self.vf_state.apply_fn({"params": params}, t, x, null_cond)
            #return (1 + self.cfg_ode_weight) * v_cond - self.cfg_ode_weight * v_uncond

        def solve_ode(x: jnp.ndarray, condition: jnp.ndarray | None) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf_cfg if self.cfg_p_resample else vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=condition,
                **kwargs,
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None]))(x, condition)
        return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
