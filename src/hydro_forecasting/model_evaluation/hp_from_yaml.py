import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import yaml
from typing import Dict, Any, Set, Tuple
import importlib


def get_config_class(model_type: str):
    """
    Dynamically import the appropriate config class based on model type.

    Args:
        model_type: The type of model (tide, tft, ealstm, tsmixer)

    Returns:
        The config class for the specified model

    Raises:
        ImportError: If the model type is not recognized
    """
    model_type = model_type.lower()
    try:
        module = importlib.import_module(
            f"src.models.{model_type}.config", package="model_evaluation"
        )

        # Handle different naming conventions for config classes:
        if model_type == "tft":
            class_name = "TFTConfig"
        elif model_type == "tide":
            class_name = "TiDEConfig"
        elif model_type == "ealstm":
            class_name = "EALSTMConfig"
        elif model_type == "tsmixer":
            class_name = "TSMixerConfig"

        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import config for model type '{model_type}': {e}")


def get_expected_params(model_type: str) -> Tuple[Set[str], Set[str]]:
    """
    Get the expected parameters for a specified model type.

    Args:
        model_type: The type of model (tide, tft, ealstm, tsmixer)

    Returns:
        Tuple containing (standard_params, model_specific_params)
    """
    try:
        config_class = get_config_class(model_type)
        standard_params = set(config_class.STANDARD_PARAMS)
        model_params = set(config_class.MODEL_PARAMS)
        return standard_params, model_params
    except ImportError as e:
        print(f"Error getting expected parameters: {e}")
        return set(), set()


def hp_from_yaml(model_type: str, yaml_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a YAML file for a specific model type.

    Args:
        model_type: Model type (tide, tft, ealstm, tsmixer)
        yaml_path: Path to the YAML file containing hyperparameters

    Returns:
        Dictionary of hyperparameters suitable for model configuration

    Raises:
        FileNotFoundError: If the YAML file does not exist
        PermissionError: If the YAML file cannot be accessed due to permissions
        yaml.YAMLError: If the YAML file has invalid syntax
    """
    # Validate inputs
    if not model_type or not isinstance(model_type, str):
        raise ValueError(f"Invalid model type: {model_type}")

    model_type = model_type.lower()
    valid_models = {"tide", "tft", "ealstm", "tsmixer"}
    if model_type not in valid_models:
        raise ValueError(
            f"Unsupported model type: {model_type}. Must be one of: {', '.join(valid_models)}"
        )

    # Check file existence and accessibility
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    if not os.path.isfile(yaml_path):
        raise ValueError(f"Path is not a file: {yaml_path}")

    if not os.access(yaml_path, os.R_OK):
        raise PermissionError(f"Cannot read YAML file (permission denied): {yaml_path}")

    # Load YAML file
    try:
        with open(yaml_path, "r") as f:
            yaml_params = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {yaml_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading YAML file {yaml_path}: {e}")

    if not yaml_params or not isinstance(yaml_params, dict):
        raise ValueError(
            f"YAML file {yaml_path} does not contain a valid parameter dictionary"
        )

    # Get expected parameters for the model
    standard_params, model_params = get_expected_params(model_type)
    all_expected_params = standard_params.union(model_params)

    # Check for missing parameters
    found_params = set(yaml_params.keys())
    missing_params = all_expected_params - found_params

    if missing_params:
        print(
            "The following parameters were not found in the YAML file and will use defaults:"
        )
        for param in sorted(missing_params):
            # Indicate which category the parameter belongs to
            category = "standard" if param in standard_params else "model-specific"
            print(f"  - {param} ({category})")

    # Check for unexpected parameters
    unexpected_params = found_params - all_expected_params
    if unexpected_params:
        print(
            f"The following parameters in the YAML file are not recognized for {model_type} model:"
        )
        for param in sorted(unexpected_params):
            print(f"  - {param}")

    return yaml_params


def load_model_config(
    model_type: str, yaml_path: str, **override_params
) -> Dict[str, Any]:
    """
    Load model configuration with hyperparameters from YAML and allow parameter overrides.

    Args:
        model_type: Model type (tide, tft, ealstm, tsmixer)
        yaml_path: Path to the YAML file containing hyperparameters
        **override_params: Additional parameters to override YAML values

    Returns:
        Complete configuration dictionary for model initialization
    """
    # Load parameters from YAML
    config_params = hp_from_yaml(model_type, yaml_path)

    # Override with any explicitly provided parameters
    for key, value in override_params.items():
        if value is not None:  # Only override if a value is provided
            config_params[key] = value
            print(f"Overriding parameter '{key}' with value: {value}")

    return config_params


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load model hyperparameters from YAML")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tide", "tft", "ealstm", "tsmixer"],
        help="Model type",
    )
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML file")
    args = parser.parse_args()

    try:
        # Load hyperparameters
        params = hp_from_yaml(args.model, args.yaml)

        # Display loaded parameters
        print(f"\nLoaded parameters for {args.model} model:")
        for key, value in sorted(params.items()):
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
