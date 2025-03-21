import functools
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.model._utils import _multivariate_normal

__all__ = ["GENOT"]

LinTerm = tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]
DataMatchFn = Callable[[LinTerm], jnp.ndarray] | Callable[[QuadTerm], jnp.ndarray]


class GENOT:
    """GENOT :cite:`klein:23` extended to the conditional setting.

    Parameters
    ----------
    vf
        Vector field parameterized by a neural network.
    flow
        Flow between the latent and the target distributions.
    data_match_fn
        Function to match samples from the source and the target
        distributions. Depending on the data passed :meth:`step_fn`, it has
        the following signature:

        - ``(src_lin, tgt_lin) -> matching`` - linear matching.
        - ``(src_quad, tgt_quad, src_lin, tgt_lin) -> matching`` - quadratic (fused) GW matching.
        In the pure GW setting, both ``src_lin`` and ``tgt_lin`` will be set to :obj:`None`.

    source_dim
        Dimensionality of the source distribution.
    target_dim
        Dimensionality of the target distribution.
    time_sampler
        Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
        :func:`ott.solvers.utils.uniform_sampler`.
    latent_noise_fn
        Function to sample from the latent distribution in the
        target space with a ``(rng, shape) -> noise`` signature.
        If :obj:`None`, multivariate normal distribution is used.
    kwargs
        Keyword arguments.
    """

    def __init__(
        self,
        vf: velocity_field.VelocityField,
        flow: dynamics.BaseFlow,
        data_match_fn: DataMatchFn,
        *,
        source_dim: int,
        target_dim: int,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        latent_noise_fn: (Callable[[jax.Array, tuple[int, ...]], jnp.ndarray] | None) = None,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.flow = flow
        self.data_match_fn = jax.jit(data_match_fn)
        self.time_sampler = time_sampler
        self.source_dim = source_dim
        if latent_noise_fn is None:
            latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim)
        self.latent_noise_fn = latent_noise_fn

        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
            **kwargs,
        )
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self) -> Callable:  #  type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            latent: jnp.ndarray,
            conditions: dict[str, jnp.ndarray] | None,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                latent: jnp.ndarray,
                condition: dict[str, jnp.ndarray] | None,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.flow.compute_xt(rng_flow, t, latent, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    source,
                    condition,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.flow.compute_ut(t, x_t, latent, target)

                flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                encoder_loss = self.condition_encoder_regularization * (
                    condition_mean_regularization + condition_var_regularization
                )
        
                return flow_matching_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, latent, conditions, rng)

            return loss, vf_state.apply_gradients(grads=grads)

        return vf_step_fn

    @staticmethod
    def _prepare_data(
        batch: dict[str, jnp.ndarray],
    ) -> tuple[
        tuple[ArrayLike, ArrayLike],
        tuple[ArrayLike | None, ...],
    ]:
        src_lin, src_quad = batch.get("src_cell_data"), batch.get("src_cell_data_quad")
        tgt_lin, tgt_quad = batch.get("tgt_cell_data"), batch.get("tgt_cell_data_quad")

        if src_quad is None and tgt_quad is None:  # lin
            src, tgt = src_lin, tgt_lin
            arrs = src_lin, tgt_lin
        elif src_lin is None and tgt_lin is None:  # quad
            src, tgt = src_quad, tgt_quad
            arrs = src_quad, tgt_quad
        elif all(arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)):  # fused quad
            src = jnp.concatenate([src_lin, src_quad], axis=1)
            tgt = jnp.concatenate([tgt_lin, tgt_quad], axis=1)
            arrs = src_quad, tgt_quad, src_lin, tgt_lin
        else:
            raise RuntimeError("Cannot infer OT problem type from data.")

        return (src, tgt), arrs  # type: ignore[return-value]

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ):
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.
        """
        rng = jax.random.split(rng, 5)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

        condition = batch.get("condition")
        (src, tgt), matching_data = self._prepare_data(batch)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        latent = self.latent_noise_fn(rng_noise, (n,))

        tmat = self.data_match_fn(*matching_data)
        src_ixs, tgt_ixs = solver_utils.sample_joint(
            rng_resample,
            tmat,
        )

        src, tgt = src[src_ixs], tgt[tgt_ixs]
        loss, self.vf_state = self.vf_step_fn(rng_step_fn, self.vf_state, time, src, tgt, latent, condition)
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.


        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def predict(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        **kwargs: Any,
    ) -> ArrayLike | tuple[ArrayLike, diffrax.Solution]:
        """Generate the push-forward of ``'source'`` under condition ``'condition'``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        source
            Input data of shape [batch_size, ...].
        condition
            Condition of the input data of shape [batch_size, ...].
        rng
            Random generate used to sample from the latent distribution.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``'x'`` under condition ``'condition'``.
        """
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs.setdefault("max_steps", 100000)

        use_mean = True if rng is None else False
        rng = utils.default_prng_key(rng) if rng is None else rng
        condition_mean, condition_logvar = self.get_condition_embedding(condition, return_as_numpy=False)

        if self.condition_encoder_mode == "deterministic" or use_mean:
            cond_embedding = condition_mean
        else:
            rng, rng_embed = jax.random.split(rng)
            cond_embedding = condition_mean + jax.random.normal(rng_embed, condition_mean.shape) * jnp.exp(
                0.5 * condition_logvar
            )

        def vf(t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
            x_0, cond_embedding = args
            params = self.vf_state.params
            t_array = jnp.array([[t]])
            y_array = y[None, :]
            preds, _, _ = self.vf.apply(
                {"params": params},
                t=t_array,
                x_t=y_array,
                x_0=x_0,
                cond_embedding=cond_embedding,
                train=False,
            )
            return preds[0]

        def solve_ode(latent: jnp.ndarray, args: tuple) -> jnp.ndarray:
            term = diffrax.ODETerm(vf)
            sol = diffrax.diffeqsolve(
                term,
                t0=0.0,
                t1=1.0,
                y0=latent,
                args=args,
                **kwargs,
            )
            return sol.ys[0]

        latent = self.latent_noise_fn(rng, (len(x),))

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=(0, (0, None))))(latent, (x[:, None], cond_embedding))
        return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """If the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value) -> None:
        self._is_trained = value
