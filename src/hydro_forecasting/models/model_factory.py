import importlib
from collections.abc import Callable
from typing import Any, cast

from ..model_evaluation.hp_from_yaml import hp_from_yaml


def _get_tide_config() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.tide", package=__package__).TiDEConfig)


def _get_tide_model() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.tide", package=__package__).LitTiDE)


def _get_tsmixer_config() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.tsmixer", package=__package__).TSMixerConfig)


def _get_tsmixer_model() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.tsmixer", package=__package__).LitTSMixer)


def _get_ealstm_config() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.ealstm", package=__package__).EALSTMConfig)


def _get_ealstm_model() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.ealstm", package=__package__).LitEALSTM)


def _get_tft_config() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.tft", package=__package__).TFTConfig)


def _get_tft_model() -> type[Any]:
    return cast(type[Any], importlib.import_module("..models.tft", package=__package__).LitTFT)


# Model Registry
# Maps model_type strings to functions that return the model and config classes.
MODEL_REGISTRY: dict[str, dict[str, Callable[[], type[Any]]]] = {
    "tide": {
        "config_class_getter": _get_tide_config,
        "model_class_getter": _get_tide_model,
    },
    "tsmixer": {
        "config_class_getter": _get_tsmixer_config,
        "model_class_getter": _get_tsmixer_model,
    },
    "ealstm": {
        "config_class_getter": _get_ealstm_config,
        "model_class_getter": _get_ealstm_model,
    },
    "tft": {
        "config_class_getter": _get_tft_config,
        "model_class_getter": _get_tft_model,
    },
}


def get_model_config_class(model_type: str) -> type[Any]:
    """
    Get the model configuration class based on the model type.

    Args:
        model_type: Type of model ('tide', 'tsmixer', etc.)

    Returns:
        Model configuration class.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_REGISTRY[model_type]["config_class_getter"]()


def get_model_class(model_type: str) -> type[Any]:
    """
    Get the model class based on the model type.

    Args:
        model_type: Type of model ('tide', 'tsmixer', etc.)

    Returns:
        Model class.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_REGISTRY[model_type]["model_class_getter"]()


def create_model(model_type: str, yaml_path: str) -> tuple[Any, dict[str, Any]]:
    """
    Create a model instance from a YAML file.

    Args:
        model_type: Type of model to create ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML file

    Returns:
        Tuple containing:
        - Model instance
        - Dictionary of model hyperparameters
    """
    model_hp = hp_from_yaml(model_type, yaml_path)

    ModelConfigClass = get_model_config_class(model_type)
    ModelClass = get_model_class(model_type)

    model_config = ModelConfigClass(**model_hp)
    model = ModelClass(config=model_config)

    return model, model_hp


def load_pretrained_model(
    model_type: str,
    yaml_path: str,
    checkpoint_path: str,
    lr_factor: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """
    Load a pretrained model from a checkpoint.

    Args:
        model_type: Type of model to load
        yaml_path: Path to model hyperparameter YAML file
        checkpoint_path: Path to model checkpoint
        lr_factor: Factor to reduce learning rate by for fine-tuning

    Returns:
        Tuple containing:
        - Loaded model instance
        - Dictionary of model hyperparameters
    """
    model_hp = hp_from_yaml(model_type, yaml_path)

    ModelConfigClass = get_model_config_class(model_type)
    ModelClass = get_model_class(model_type)

    model_config = ModelConfigClass(**model_hp)
    model = ModelClass.load_from_checkpoint(checkpoint_path, config=model_config)

    # Ensure hparams exists and has learning_rate, common in PyTorch Lightning
    if hasattr(model, "hparams") and "learning_rate" in model.hparams:
        original_lr = model.hparams.learning_rate
        new_lr = original_lr / lr_factor
        model.hparams.learning_rate = new_lr

        model.original_lr = original_lr
        model.fine_tuned_lr = new_lr

        model_hp["learning_rate"] = new_lr
        model_hp["original_lr"] = original_lr
    else:
        print(
            f"Warning: Could not adjust learning rate for model_type {model_type}. "
            "Model may not have 'hparams' with 'learning_rate'."
        )

    return model, model_hp


def create_model_from_config_dict(
    model_type: str,
    config_dict: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Create a model instance from a configuration dictionary.

    Args:
        model_type: Type of model to create
        config_dict: Dictionary of model hyperparameters

    Returns:
        Tuple containing:
        - Model instance
        - Dictionary of model hyperparameters
    """
    ModelConfigClass = get_model_config_class(model_type)
    ModelClass = get_model_class(model_type)
    model_config = ModelConfigClass(**config_dict)
    model = ModelClass(config=model_config)
    return model, config_dict
