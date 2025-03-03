from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from cfp._types import ArrayLike

__all__ = [
    "BaseDataMixin",
    "ConditionData",
    "PredictionData",
    "TrainingData",
    "ValidationData",
]


@dataclass
class ReturnData:  # TODO: this should rather be a NamedTuple
    split_covariates_mask: np.ndarray | None
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_covariates_mask: np.ndarray | None
    perturbation_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]
    control_to_perturbation: dict[int, ArrayLike]
    max_combination_length: int


class ArrayConversionMixin:
    """Mixin for converting numpy arrays to jax arrays."""

    def __post_init__(self, convert_to_jax: bool = True) -> None:
        """Convert numpy arrays to jax arrays after initialization.

        Parameters
        ----------
        convert_to_jax
            Whether to convert numpy arrays to jax arrays.
        """
        if not convert_to_jax:
            return

        # Convert array attributes to jax arrays
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, np.ndarray):
                setattr(self, attr_name, jnp.array(attr_value))
            elif isinstance(attr_value, dict) and attr_name == "condition_data":
                # Handle condition_data dictionary specifically
                for key, val in attr_value.items():
                    if isinstance(val, np.ndarray):
                        attr_value[key] = jnp.array(val)
            elif isinstance(attr_value, dict) and attr_name == "control_to_perturbation":
                # Handle control_to_perturbation dictionary
                for key, val in attr_value.items():
                    if isinstance(val, np.ndarray):
                        attr_value[key] = jnp.array(val)


class BaseDataMixin:
    """Base class for data containers."""

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbations(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbation_covariates(self) -> int:
        """Returns the number of perturbation covariates."""
        return len(self.condition_data)  # type: ignore[attr-defined]

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


@dataclass
class ConditionData(BaseDataMixin, ArrayConversionMixin):
    """Data container containing condition embeddings.

    Parameters
    ----------
    condition_data
        Dictionary with embeddings for conditions.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    data_manager
        Data manager used to generate the data.
    convert_to_jax
        Whether to convert numpy arrays to jax arrays.
    """

    condition_data: dict[str, ArrayLike]
    max_combination_length: int
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    perturbation_idx_to_id: dict[int, Any]
    null_value: Any
    data_manager: Any
    convert_to_jax: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__(self.convert_to_jax)


@dataclass
class TrainingData(BaseDataMixin, ArrayConversionMixin):
    """Training data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in ``split_covariates_mask``.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in ``perturbation_covariates_mask``.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
    convert_to_jax
        Whether to convert numpy arrays to jax arrays.
    """

    cell_data: ArrayLike  # (n_cells, n_features)
    split_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, ArrayLike]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    null_value: Any
    data_manager: Any
    convert_to_jax: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__(self.convert_to_jax)


@dataclass
class ValidationData(BaseDataMixin, ArrayConversionMixin):
    """Data container for the validation data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in ``split_covariates_mask``.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in ``perturbation_covariates_mask``.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
    n_conditions_on_log_iteration
        Number of conditions to use for computation callbacks at each logged iteration.
        If :obj:`None`, use all conditions.
    n_conditions_on_train_end
        Number of conditions to use for computation callbacks at the end of training.
        If :obj:`None`, use all conditions.
    convert_to_jax
        Whether to convert numpy arrays to jax arrays.
    """

    cell_data: ArrayLike  # (n_cells, n_features)
    split_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, ArrayLike]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    null_value: Any
    data_manager: Any
    n_conditions_on_log_iteration: int | None = None
    n_conditions_on_train_end: int | None = None
    convert_to_jax: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__(self.convert_to_jax)


@dataclass
class PredictionData(BaseDataMixin, ArrayConversionMixin):
    """Data container to perform prediction.

    Parameters
    ----------
    src_data
        Dictionary with data for source cells.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    covariate_encoder
        Encoder for the primary covariate.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking ``null_value``.
    convert_to_jax
        Whether to convert numpy arrays to jax arrays.
    """

    cell_data: ArrayLike  # (n_cells, n_features)
    split_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, ArrayLike]
    max_combination_length: int
    null_value: Any
    data_manager: Any
    convert_to_jax: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__(self.convert_to_jax)
