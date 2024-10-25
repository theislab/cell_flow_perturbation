import abc
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen import initializers
from flax.training.train_state import TrainState
from flax.typing import FrozenDict

from cfp._batch_norm import BNTrainState
from cfp._constants import GENOT_CELL_KEY
from cfp._types import ArrayLike, Layers_separate_input_t, Layers_t
from cfp._distributions import _multivariate_normal
from cfp.networks._set_encoders import MLPBlock, BaseModule

__all__ = [
    "CountsEncoder",
    "CountsDecoder",
]


class CountsEncoder(BaseModule):
    """
    Implements the AutoEncoder architecture of the CountsAE model

    Parameters
    ----------
    input_dim: dict[str, int]
        The dimensionality of the input for each modality
    encoder_kwargs: dict[str, dict[str, Any]]
        The key-word arguments for initializing the encoder (one for each modality)
    covariate_specific_theta: bool
        Whether to use a dispersion coefficient for each covariate
    conditioning_covariate: str
        Literal indicating the covariate used for conditioning
    n_cat: int
        The number of unique categorical variables
    """

    input_dim: int
    encoder_kwargs: dict[str, Any]
    covariate_specific_theta: bool
    n_cat: int | None

    """Implements the Encoder block for count data"""

    def setup(self) -> None:
        """Initialize the module."""
        encoder_kwargs = self.encoder_kwargs.unfreeze()
        encoder = MLPBlock(**encoder_kwargs)
        self.encoder = encoder

    def __call__(
        self, X: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        """Encodes the Input"""
        z = self.encoder(X, training=training)
        return z

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent space"""
        return self.encoder_kwargs["dims"][-1]

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        training: bool = False,
    ) -> BNTrainState | TrainState:
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
        # computing dummy counts
        x = jnp.ones((1, self.input_dim))
        # retrieving the state of the module
        variables = self.init(rng, x, training=training)
        params = variables["params"]
        if self.encoder_kwargs["batch_norm"]:
            batch_stats = variables["batch_stats"]
            return BNTrainState.create(
                apply_fn=self.apply,
                params=params,
                tx=optimizer,
                batch_stats=batch_stats,
            )
        else:
            return TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class CountsDecoder(BaseModule):
    """
    Implements the AutoEncoder architecture of the CountsAE model

    Parameters
    ----------
    input_dim: dict[str, int]
        The dimensionality of the input for each modality
    encoder_kwargs: dict[str, dict[str, Any]]
        The key-word arguments for initializing the encoder (one for each modality)
    covariate_specific_theta: bool
        Whether to use a dispersion coefficient for each covariate
    conditioning_covariate: str
        Literal indicating the covariate used for conditioning
    n_cat: int
        The number of unique categorical variables
    is_binarized: bool
        Whether the input data is binarized
    """

    input_dim: int
    encoder_kwargs: dict[str, Any]
    covariate_specific_theta: bool
    n_cat: int | None

    """Implements the Encoder block of the CountsAE model"""

    def setup(self) -> None:
        """Initialize the module."""

        # copying the encoder kwargs attribures to modify it safely
        encoder_kwargs = self.encoder_kwargs.unfreeze()
        encoder_kwargs["dims"] = [*encoder_kwargs["dims"][::-1], self.input_dim]
        decoder = MLPBlock(**encoder_kwargs)
        self.decoder = decoder
        ## theta
        if not self.covariate_specific_theta:
            shape = 1
        else:
            shape = self.n_cat
        self.param(
            "theta",
            _multivariate_normal,
            shape=shape,
            dim=self.input_dim,
            mean=0.0,
            cov=1.0,
        )

    def __call__(
        self, z: jnp.ndarray, size_factor: jnp.ndarray, training: bool = True
    ) -> dict[str, jnp.ndarray]:
        """Encodes the Input"""
        modality_list = list(self.encoder_kwargs.keys())
        x_hat = self.decoder(z, training)
        x_hat = nn.softmax(x_hat, axis=1)
        mu_hat = x_hat * size_factor
        return mu_hat

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent space"""
        return self.encoder_kwargs["dims"][-1]

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        training: bool = False,
    ) -> BNTrainState | TrainState:
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
        # computing dummy counts and size factor
        x = jnp.ones((1, self.input_dim))
        size_factor = jnp.sum(x, axis=1, keepdims=True)
        # computing dummy latent state
        z = jnp.ones((1, self.latent_dim))
        # retrieving the state of the module
        variables = self.init(rng, z, size_factor, training=training)
        params = variables["params"]
        if self.encoder_kwargs["batch_norm"]:
            batch_stats = variables["batch_stats"]
            return BNTrainState.create(
                apply_fn=self.apply,
                params=params,
                tx=optimizer,
                batch_stats=batch_stats,
            )
        else:
            return TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
