import abc
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen import initializers
from flax.training import train_state
from flax.typing import FrozenDict

from cfp._constants import GENOT_CELL_KEY
from cfp._types import ArrayLike, Layers_separate_input_t, Layers_t
from cfp._distributions import _multivariate_normal
from cfp.networks._set_encoders import MLPBlock, BaseModule

__all__ = [
    "CFGenEncoder",
    "CFGenDecoder",
    "CFGenAE"
]


class CFGenEncoder(BaseModule):
    """
    Implements the AutoEncoder architecture of the CFGen model

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
    input_dim: dict[str, int]
    encoder_kwargs: dict[str, dict[str, Any]]
    covariate_specific_theta: bool
    conditioning_covariate: str
    n_cat: int
    is_binarized: bool
    encoder_multimodal_joint_layers: dict[str, Any] | None

    """Implements the Encoder block of the CFGen model"""
    def setup(self) -> None:
        """Initialize the module."""
        
        # copying the encoder kwargs attribures to modify it safely
        encoder_kwargs = self.encoder_kwargs.unfreeze()
        # List of modalities present in the data 
        self.modality_list = list(encoder_kwargs.keys())

        # initializing the modality specific components
        encoder = dict()
        for mod in self.modality_list:
            encoder_kwargs[mod]["dims"] = [self.input_dim[mod], *self.encoder_kwargs[mod]["dims"]]
            encoder[mod] = MLPBlock(**encoder_kwargs[mod])
        self.encoder = encoder
        # initializing the joint multimodal encoder
        if self.encoder_multimodal_joint_layers:
            joint_inputs = sum([encode_kwargs[mod]["dims"][-1] for mod in self.modality_list])
            self.encoder_multimodal_joint_layers["dims"] = [joint_inputs, *self.encoder_multimodal_joint_layers]
            self.encoder_joint = MLPBlock(**self.encoder_multimodal_joint_layers)

    def __call__(
            self, 
            X: dict[str, jnp.ndarray] | jnp.ndarray,
            training: bool
        ) -> dict[str, jnp.ndarray] | jnp.ndarray:
        """Encodes the Input"""
        z = {}
        for mod in self.modality_list:
            z_mod = self.encoder[mod](X[mod], training = training)
            z[mod] = z_mod
        
        # Implement joint layers if defined
        if self.encoder_multimodal_joint_layers:
            z_joint = jnp.concatenate([z[mod] for mod in z], axis=-1)
            z = self.encoder_joint(z_joint, training = training)     
        return z

class CFGenDecoder(BaseModule):
    """
    Implements the AutoEncoder architecture of the CFGen model

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
    #input_dim: dict[str, int]
    encoder_kwargs: dict[str, dict[str, Any]]
    covariate_specific_theta: bool
    conditioning_covariate: str
    n_cat: int
    is_binarized: bool
    encoder_multimodal_joint_layers: dict[str, Any] | None

    """Implements the Encoder block of the CFGen model"""
    def setup(self) -> None:
        """Initialize the module."""
        
        # copying the encoder kwargs attribures to modify it safely
        encoder_kwargs = self.encoder_kwargs.unfreeze()
        # List of modalities present in the data 
        self.modality_list = list(encoder_kwargs.keys())
        # initializing the modality specific components
        decoder = {}
        for mod in self.modality_list:
            encoder_kwargs[mod]["dims"] = [self.input_dim[mod], *self.encoder_kwargs[mod]["dims"]]
            if self.encoder_multimodal_joint_layers:
                encoder_kwargs[mod]["dims"].append(self.encoder_multimodal_joint_layers["dims"][-1])
            encoder_kwargs[mod]["dims"] = encoder_kwargs[mod]["dims"][::-1]
            decoder[mod] = MLPBlock(**encoder_kwargs[mod])
        self.decoder = decoder
    
    def __call__(
            self, 
            z: jnp.ndarray,
            size_factor: jnp.ndarray,
            training: bool
        ) -> dict[str, jnp.ndarray]:
        """Encodes the Input"""
        mu_hat = {}
        for mod in self.modality_list:
            if not self.encoder_multimodal_joint_layers:
                x_mod = self.decoder[mod](z[mod], training)
            else:
                x_mod = self.decoder[mod](z, training)

            if mod != "atac" or (mod == "atac" and not self.is_binarized):
                mu_hat_mod = nn.softmax(x_mod, axis=1)  # for Poisson counts the parameterization is similar to RNA 
                mu_hat_mod = mu_hat_mod * size_factor[mod]
            else:
                mu_hat_mod = nn.sigmoid(x_mod)
            mu_hat[mod] = mu_hat_mod
        return mu_hat

class CFGenAE:
    """
    Implements the AutoEncoder architecture of the CFGen model

    Parameters
    ----------
    input_dim: dict[str, int]
        The dimensionality of the input for each modality
    encoder_kwargs: dict[str, dict[str, Any]]
        The key-word arguments for initializing the encoder (one for each modality)
    covariate_specific_theta: bool
        Whether to use a dispersion coefficient for each covariate
    is_binarized: bool
        Whether the input data is binarized 
    encoder_multimodal_joint_layers: dict[str, Any] | None
        The layers for the optional joint encoer across multiple modalities
    """
    input_dim: dict[str, int]
    encoder_kwargs: dict[str, dict[str, Any]]
    covariate_specific_theta: bool
    is_binarized: bool
    encoder_multimodal_joint_layers: dict[str, Any] | None

    def setup(
            self,
        ) -> None:
        """"""
        self.modality_list = list(self.encoder_kwargs.keys())
        # Learnable Theta Parameter
        in_dim_rna = self.input_dim["rna"]
        if not self.covariate_specific_theta:
            shape = (in_dim_rna)
        else:
            shape = (n_cat, in_dim_rna)
        self.param('theta', multivariate_normal, shape = shape, dim = in_dim_rna, mean = 0.0, cov = 1.0)
        
        self.encoder = CFGenEncoder( 
            input_dim = self.input_dim,
            encoder_kwargs = self.encoder_kwargs,
            covariate_specific_theta = self.covariate_specific_theta,
            is_binarized = self.is_binarized,
            encoder_multimodal_joint_layers = self.encoder_multimodal_joint_layers
        )
        self.decoder = CFGenDecoder(
            input_dim = self.input_dim,
            encoder_kwargs = self.encoder_kwargs,
            covariate_specific_theta = self.covariate_specific_theta,
            is_binarized = self.is_binarized,
            encoder_multimodal_joint_layers = self.encoder_multimodal_joint_layers
        )
    
    def __call__(
            self,
            x: dict[str, jnp.ndarray],
            training: bool   
        ) -> dict[str, jnp.ndarray]:
        ## retrieving size factors
        size_factor = {}
        for mod in self.modality_list:
            size_factor[mod] = jnp.sum(x[mod], axis = 1, keepdims = True)
        z = self.encoder(x, training)
        x_hat = self.decoder(z, size_factor, training)
        return x_hat