from collections.abc import Callable
from typing import Any, Literal

import jax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

from flax.training.train_state import TrainState
from cfp._batch_norm import BNTrainState
from cfp.external._scvi import NegativeBinomial
from cfp.networks._cfgen_ae import CFGenEncoder, CFGenDecoder
from cfp._counts import normalize_expression
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
        normalization_type: Literal[
            "none", "proportions", "log_gexp", "log_gexp_scaled"
        ] = "none",
    ):
        self._is_trained: bool = False
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_state = self.encoder.create_train_state(
            input_dim=self.encoder.input_dim, **kwargs_encoder
        )
        self.decoder_state = self.decoder.create_train_state(
            input_dim=self.encoder.encoder_kwargs["dims"][-1], **kwargs_decoder
        )
        self.normalization_type = normalization_type
        self.ae_fwd_fn = self._get_ae_fwd_fn()

    def _get_ae_fwd_fn(self) -> Callable:
        @jax.jit
        def ae_fwd_fn(
            encoder_state: TrainState | BNTrainState,
            decoder_state: TrainState | BNTrainState,
            counts: jnp.ndarray,
        ) -> tuple[Any, Any, Any]:
            def loss_fn(
                encoder_params: jnp.ndarray,
                decoder_params: jnp.ndarray,
                counts: jnp.ndarray,
            ) -> float:
                size_factor = jnp.sum(counts, axis=1, keepdims=True)
                normalized_counts = normalize_expression(
                    counts, size_factor, self.normalization_type
                )
                ## defining the parameter dictionaries for encoder and decoder blocks
                ## we need to modify it later in case we are using batch normalization
                encoder_params_dict = {"params": encoder_params}
                decoder_params_dict = {"params": decoder_params}
                ## setting the default value for the `mutable` argument of `apply_fn` 
                mutable_enc = False
                mutable_dec = False
                ## updating the encoder and decoder parameters in case of batch normalization
                if self.encoder.encoder_kwargs["batch_norm"]:
                    encoder_params_dict["batch_stats"] = encoder_state.batch_stats
                    mutable_enc = ["batch_stats"]
                if self.decoder.encoder_kwargs["batch_norm"]:
                    decoder_params_dict["batch_stats"] = decoder_state.batch_stats
                    mutable_dec = ["batch_stats"]
                ## forward pass on the encoder
                ## in case of batch normalization retrieving the update of the stats 
                encoder_out = encoder_state.apply_fn(
                    encoder_params_dict, normalized_counts, training = True, mutable = mutable_enc
                )
                if self.encoder.encoder_kwargs["batch_norm"]:
                    z, enc_updates = encoder_out
                else:
                    z = encoder_out
                ## forward pass on the decoder
                ## in case of batch normalization retrieving the update of the stats
                decoder_out = decoder_state.apply_fn(
                    decoder_params_dict, z, size_factor, training = True, mutable = mutable_dec
                )
                if self.decoder.encoder_kwargs["batch_norm"]:
                    x_hat, dec_updates = decoder_out
                else:
                    x_hat = decoder_out
                ## computing reconstruction loss
                px = NegativeBinomial(
                    mean=x_hat,
                    inverse_dispersion=jnp.exp(decoder_params["theta"]),
                )
                loss = -px.log_prob(counts).sum(1).mean()
                if not self.encoder.encoder_kwargs["batch_norm"]:
                    return loss
                else:
                    return loss, (enc_updates, dec_updates)

            grad_fn = jax.value_and_grad(
                loss_fn,
                argnums=(0, 1),
                has_aux=self.encoder.encoder_kwargs["batch_norm"],
            )
            loss_step, (encoder_grads, decoder_grads) = grad_fn(
                encoder_state.params,
                decoder_state.params,
                counts,
            )
            if not self.encoder.encoder_kwargs["batch_norm"]:
                loss = loss_step
                return (
                    encoder_state.apply_gradients(grads=encoder_grads),
                    decoder_state.apply_gradients(grads=decoder_grads),
                    loss,
                )
            else:
                loss, (enc_updates, dec_updates) = loss_step
                return (
                    encoder_state.apply_gradients(
                        grads=encoder_grads, batch_stats=enc_updates["batch_stats"]
                    ),
                    decoder_state.apply_gradients(
                        grads=decoder_grads, batch_stats=dec_updates["batch_stats"]
                    ),
                    loss,
                )

        return ae_fwd_fn

    def fwd_fn(self, rng: np.ndarray, batch: dict[str, ArrayLike], training: bool):
        counts = batch["src_cell_data"]
        self.encoder_state, self.decoder_state, loss = self.ae_fwd_fn(
            self.encoder_state,
            self.decoder_state,
            counts,
        )
        return loss

    def predict(self, counts: ArrayLike, training: bool):
        """"""
        size_factor = jnp.sum(counts, axis=1, keepdims=True)
        normalized_counts = normalize_expression(
            counts, size_factor, self.normalization_type
        )
        ## defining the parameter dictionaries for encoder and decoder blocks
        ## we need to modify it later in case we are using batch normalization
        encoder_params_dict = {"params": self.encoder_state.params}
        decoder_params_dict = {"params": self.decoder_state.params}
        ## setting the default value for the `mutable` argument of `apply_fn` 
        mutable_enc = False
        mutable_dec = False
        ## updating the encoder and decoder parameters in case of batch normalization
        if self.encoder.encoder_kwargs["batch_norm"]:
            encoder_params_dict["batch_stats"] = self.encoder_state.batch_stats
            mutable_enc = ["batch_stats"]
        if self.decoder.encoder_kwargs["batch_norm"]:
            decoder_params_dict["batch_stats"] = self.decoder_state.batch_stats
            mutable_dec = ["batch_stats"]
        ## forward pass on the encoder
        ## in case of batch normalization retrieving the update of the stats 
        encoder_out = self.encoder_state.apply_fn(
            encoder_params_dict, normalized_counts, training = training, mutable = mutable_enc
        )
        if self.encoder.encoder_kwargs["batch_norm"]:
            z, enc_updates = encoder_out
        else:
            z = encoder_out
        ## forward pass on the decoder
        ## in case of batch normalization retrieving the update of the stats
        decoder_out = self.decoder_state.apply_fn(
            decoder_params_dict, z, size_factor, training = training, mutable = mutable_dec
        )
        if self.decoder.encoder_kwargs["batch_norm"]:
            x_hat, dec_updates = decoder_out
        else:
            x_hat = decoder_out
        return x_hat, self.decoder_state.params["theta"]

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
