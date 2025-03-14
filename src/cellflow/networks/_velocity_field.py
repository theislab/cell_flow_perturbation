from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from ott.neural.networks.layers import time_encoder

from cellflow._constants import GENOT_CELL_KEY
from cellflow._logging import logger
from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder, MLPBlock

__all__ = ["ConditionalVelocityField"]


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        encode_conditions
                Processes the embedding of the perturbation conditions if :obj:`True`. If
                :obj:`False`, directly inputs the embedding of the perturbation conditions to the
                generative velocity field. In the latter case, ``'condition_embedding_dim'``,
                ``'condition_encoder_kwargs'``, ``'pooling'``, ``'pooling_kwargs'``,
                ``'layers_before_pool'``, ``'layers_after_pool'``, ``'cond_output_dropout'``
                are ignored.
        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

    output_dim: int
    max_combination_length: int
    encode_conditions: bool = True
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    def setup(self):
        """Initialize the network."""
        if self.encode_conditions:
            self.condition_encoder = ConditionEncoder(
                output_dim=self.condition_embedding_dim,
                pooling=self.pooling,
                pooling_kwargs=self.pooling_kwargs,
                layers_before_pool=self.layers_before_pool,
                layers_after_pool=self.layers_after_pool,
                output_dropout=self.cond_output_dropout,
                covariates_not_pooled=self.covariates_not_pooled,
                mask_value=self.mask_value,
                **self.condition_encoder_kwargs,
            )

        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

    def __call__(
        self,
        rng: jnp.ndarray,
        t: jnp.ndarray,
        x: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through the neural vector field.

        Parameters
        ----------
            rng
                Random number generator.
            t
                Time of shape ``[batch, 1]``.
            x
                Data of shape ``[batch, ...]``.
            condition
                Condition dictionary, with condition names as keys and condition representations
                of shape ``[batch, max_combination_length, condition_dim]`` as values.
            train
                If :obj:`True`, enables dropout for training.

        Returns
        -------
            Output of the neural vector field of shape ``[batch, output_dim]``.
        """
        squeeze = x.ndim == 1
        if self.encode_conditions:
            cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        else:
            cond_mean = jnp.concatenate(list(cond.values()), axis=-1)
            cond_logvar = jnp.zeros_like(cond)
        if self.condition_encoder.mode == "deterministic":
            cond = cond_mean
        else:
            cond = cond_mean + jax.random.normal(rng, cond_mean.shape) * jnp.exp(0.5 * cond_logvar)
        t = time_encoder.cyclical_time_encoder(t, n_freqs=self.time_freqs)
        t = self.time_encoder(t, training=train)
        x = self.x_encoder(x, training=train)
        if squeeze:
            cond = jnp.squeeze(cond)  # , 0)
        elif cond.shape[0] != x.shape[0]:  # type: ignore[attr-defined]
            cond = jnp.tile(cond, (x.shape[0], 1))

        t = self.layer_norm_time(t)
        x = self.layer_norm_x(x)
        cond = self.layer_norm_condition(cond)

        concatenated = jnp.concatenate((t, x, cond), axis=-1)
        out = self.decoder(concatenated, training=train)
        return self.output_layer(out), cond_mean, cond_logvar

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Get the embedding of the condition.

        Parameters
        ----------
            condition
                Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
            Embedding of the condition.
        """
        if self.encode_conditions:
            condition = self.condition_encoder(condition, training=False, return_conditions_only=True)
        else:
            condition = jnp.concatenate(list(condition.values()), axis=-1)
            logger.warning("Condition encoder is not defined. Returning concatenated input as the embedding.")
        return condition

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
        additional_cond_dim: int = 0,
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
            rng
                Random number generator.
            optimizer
                Optimizer.
            input_dim
                Dimensionality of the velocity field.

        Returns
        -------
            The training state.
        """
        t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        if additional_cond_dim:
            cond[GENOT_CELL_KEY] = jnp.ones((1, additional_cond_dim))
        params = self.init(rng, t, x, cond, train=False)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    @property
    def output_dims(self):
        """Dimensions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        """The time encoder used."""
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        """Set the time encoder."""
        self._time_encoder = encoder

    @property
    def x_encoder(self):
        """The x encoder used."""
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        """Set the x encoder."""
        self._x_encoder = encoder

    @property
    def decoder(self):
        """The decoder used."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder."""
        self._decoder = decoder
