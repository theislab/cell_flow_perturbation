from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from cfp._batch_norm import BNTrainState
from cfp._distributions import _multivariate_normal
from cfp.networks._set_encoders import BaseModule, MLPBlock

__all__ = [
    "CountsEncoder",
    "CountsDecoder",
]


class CountsEncoder(BaseModule):
    """
    Implements the Encoder architecture for the CountsAE model in the unimodal case

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input for the RNA expression (i.e.: number of genes).
    encoder_kwargs: dict[str, Any]
        The key-word arguments for initializing the encoder MLP block.
    covariate_specific_theta: bool
        Whether to use a dispersion coefficient for each conditioning covariate. Defaults to `False`.
    n_cat: int | None
        The number of unique categorical variables. Defaults to `False`.

    Returns
    -------
        The latent representation of RNA expression counts.
    """

    input_dim: int
    encoder_kwargs: dict[str, Any]
    covariate_specific_theta: bool
    n_cat: int | None

    def setup(self) -> None:
        """Initialize the module."""
        encoder_kwargs = self.encoder_kwargs.unfreeze()
        encoder = MLPBlock(**encoder_kwargs)
        self.encoder = encoder

    def __call__(self, X: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass through the encoder.
        Parameters
        ----------
            x
                Data of shape ``[batch, self.input_dim]``.
            training
                If :obj:`True`, enables dropout for training.

        Returns
        -------
            Latent representation of RNA expression counts of shape ``[batch, self.latent_dim]``.
        """
        z = self.encoder(X, training=training)
        return z

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent space"""
        return self.encoder_kwargs["dims"][-1]

    @property
    def uses_batch_norm(self) -> bool:
        """Whether the module is using batch normalization"""
        return self.encoder_kwargs["batch_norm"]

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
            training
                If :obj:`True`, enables dropout for training. Defaults to `True`

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
    Implements the Decoder architecture for the CountsAE model in the unimodal case

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input for the RNA expression (i.e.: number of genes).
    encoder_kwargs: dict[str, Any]
        The key-word arguments for initializing the encoder MLP block.
    covariate_specific_theta: bool
        Whether to use a dispersion coefficient for each conditioning covariate. Defaults to `False`.
    n_cat: int | None
        The number of unique categorical variables. Defaults to `False`.

    Returns
    -------
        The parameters for the Negative Binomial noise model for RNA expression counts.
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
        """Forward pass through the decoder.
        Parameters
        ----------
            z
                Latent states of shape ``[batch, self.latent_dim]``.
            size_factor
                Size factor for each cell of shape ``[batch, self.input_dim]``.
            training
                If :obj:`True`, enables dropout for training.

        Returns
        -------
            The mean of the Negative Binomial noise model for RNA expression counts ``[batch, self.latent_dim]``.
        """
        x_hat = self.decoder(z, training)
        x_hat = nn.softmax(x_hat, axis=1)
        mu_hat = x_hat * size_factor
        return mu_hat

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent space"""
        return self.encoder_kwargs["dims"][-1]

    @property
    def uses_batch_norm(self) -> bool:
        """Whether the module is using batch normalization"""
        return self.encoder_kwargs["batch_norm"]

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
            training
                If :obj:`True`, enables dropout for training. Defaults to `True`

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
