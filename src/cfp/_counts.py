import jax.numpy as jnp

from typing import Literal
from cfp._types import ArrayLike

def normalize_expression(
        X: ArrayLike, 
        size_factor: ArrayLike, 
        normalization_type: Literal["proportions", "log_gexp", "log_gexp_scaled"]
    ) -> ArrayLike:
    """Normalize gene expression data based on the specified encoder type.

    Args:
        X (torch.Tensor): Input gene expression matrix.
        size_factor (torch.Tensor): Size factors for normalization.
        normalization_type (str): Type of encoder for normalization. It can be one of the following:
                            - "proportions": Normalize by dividing by size factor.
                            - "log_gexp": Apply log transformation to gene expression data.
                            - "learnt_encoder": Apply log transformation to gene expression data.
                            - "learnt_autoencoder": Apply log transformation to gene expression data.
                            - "log_gexp_scaled": Apply log transformation after scaling by size factor.

    Returns:
        torch.Tensor: Normalized gene expression data.

    Raises:
        NotImplementedError: If the encoder type is not recognized.
    """
    if normalization_type == "proportions":
        X = X / size_factor
    elif normalization_type == "log_gexp":
        X = jnp.log1p(X)
    elif normalization_type == "log_gexp_scaled":
        X = jnp.log1p(X / size_factor)
    else:
        raise NotImplementedError(f"Encoder type '{normalization_type}' is not implemented.")
    return X