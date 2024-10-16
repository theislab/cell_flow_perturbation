from collections.abc import Callable
from typing import Any

import jax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

from flax.training import train_state
from cfp.external._scvi import NegativeBinomial
from cfp.networks._cfgen_ae import CFGenEncoder, CFGenDecoder
from cfp._types import ArrayLike

__all__ = ["CFGen"]


class CFGen:
    """"""

    def __init__(
        self,
        encoder: CFGenEncoder,
        decoder: CFGenDecoder,
        kwargs_encoder: dict[str, Any] = {},
        kwargs_decoder: dict[str, Any] = {},
    ):
        self._is_trained: bool = False
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_state = self.encoder.create_train_state(
            input_dim=self.encoder.input_dim["rna"], **kwargs_encoder
        )
        self.decoder_state = self.decoder.create_train_state(
            input_dim=self.encoder.encoder_kwargs["rna"]["dims"][-1], **kwargs_decoder
        )
        self.ae_fwd_fn = self._get_ae_fwd_fn()

    def _get_ae_fwd_fn(self) -> Callable:
        @jax.jit
        def ae_fwd_fn(
            encoder_state: train_state.TrainState,
            decoder_state: train_state.TrainState,
            source: jnp.ndarray,
            tgt_counts: jnp.ndarray,
        ) -> tuple[Any, Any, Any]:
            def loss_fn(
                encoder_params: jnp.ndarray,
                decoder_params: jnp.ndarray,
                source: jnp.ndarray,
                tgt_counts: jnp.ndarray,
            ) -> float:
                size_factor = jnp.sum(source, axis=1, keepdims=True)
                z = encoder_state.apply_fn({"params": encoder_params}, {"rna": source})
                x_hat = decoder_state.apply_fn(
                    {"params": decoder_params}, z, {"rna": size_factor}
                )
                px = NegativeBinomial(
                    mean=x_hat["rna"],
                    inverse_dispersion=jnp.exp(decoder_params["theta"]),
                )
                loss = -px.log_prob(tgt_counts).sum(1).mean()
                return loss

            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
            loss, (encoder_grads, decoder_grads) = grad_fn(
                encoder_state.params,
                decoder_state.params,
                source,
                tgt_counts,
            )
            return (
                encoder_state.apply_gradients(grads=encoder_grads),
                decoder_state.apply_gradients(grads=decoder_grads),
                loss,
            )

        return ae_fwd_fn

    def fwd_fn(self, rng: np.ndarray, batch: dict[str, ArrayLike], training: bool):
        src = batch["src_cell_data"]
        tgt_counts = batch["src_cell_data"]  # batch["tgt_counts"]
        self.encoder_state, self.decoder_state, loss = self.ae_fwd_fn(
            self.encoder_state,
            self.decoder_state,
            src,
            tgt_counts,
        )
        return loss

    def predict(self, src: ArrayLike, training: bool):
        """"""
        size_factor = jnp.sum(src, axis=1, keepdims=True)
        z = self.encoder.apply(
            {"params": self.encoder_state.params}, {"rna": src}, False
        )
        x_hat = self.decoder.apply(
            {"params": self.decoder_state.params}, z, {"rna": size_factor}, False
        )
        return x_hat["rna"], self.decoder_state.params["theta"]

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
