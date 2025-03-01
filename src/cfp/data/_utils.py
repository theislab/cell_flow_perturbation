from collections.abc import Iterable
from typing import Any
import numpy as np
from cfp._types import ArrayLike


def _to_list(x: list[Any] | tuple[Any] | Any) -> list[Any] | tuple[Any]:
    """Converts x to a list if it is not already a list or tuple."""
    if isinstance(x, (list | tuple)):
        return x
    return [x]


def _flatten_list(x: Iterable[Iterable[Any]]) -> list[Any]:
    """Flattens a list of lists."""
    return [item for sublist in x for item in sublist]


def _check_shape(arr: float | ArrayLike) -> ArrayLike:
    if not hasattr(arr, "shape") or len(arr.shape) == 0:
        return np.ones((1, 1)) * arr
    if arr.ndim == 1:  # type: ignore[union-attr]
        return np.expand_dims(arr, 0)
    elif arr.ndim == 2:  # type: ignore[union-attr]
        if arr.shape[0] == 1:
            return arr  # type: ignore[return-value]
        if arr.shape[1] == 1:
            return np.transpose(arr)
        raise ValueError(
            "Condition representation has an unexpected shape. Should be (1, n_features) or (n_features, )."
        )
    elif arr.ndim > 2:  # type: ignore[union-attr]
        raise ValueError("Condition representation has too many dimensions. Should be 1 or 2.")

    raise ValueError(
        "Condition representation as an unexpected format. Expected an array of shape (1, n_features) or (n_features, )."
    )


def _pad_to_max_length(arr: np.ndarray, max_combination_length: int, null_value: Any) -> np.ndarray:
    if arr.shape[0] < max_combination_length:
        null_arr = np.full((max_combination_length - arr.shape[0], arr.shape[1]), null_value)
        arr = np.concatenate([arr, null_arr], axis=0)
    return arr
