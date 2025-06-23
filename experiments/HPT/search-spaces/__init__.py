"""Hyperparameter search space definitions for hydrological models."""

from typing import Any, Dict

from .ealstm_space import get_ealstm_space
from .tft_space import get_tft_space
from .tide_space import get_tide_space
from .tsmixer_space import get_tsmixer_space


def get_hyperparameter_space(model_type: str) -> dict[str, dict[str, Any]]:
    """
    Get the hyperparameter search space for the specified model type.

    Args:
        model_type: Type of model ('tide', 'tsmixer', 'ealstm', 'tft')

    Returns:
        Dictionary defining the hyperparameter search space

    Raises:
        ValueError: If the model type is not supported
    """
    model_type = model_type.lower()

    if model_type == "tide":
        return get_tide_space()
    elif model_type == "tsmixer":
        return get_tsmixer_space()
    elif model_type == "ealstm":
        return get_ealstm_space()
    elif model_type == "tft":
        return get_tft_space()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
