import hashlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from ..exceptions import ConfigurationError, FileOperationError

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from ..data_deprecated.lazy_datamodule import HydroLazyDataModule
    from ..preprocessing.grouped import GroupedPipeline

# Fixed namespace for hydro-forecasting processing runs
PROCESSING_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def _make_hashable(obj: Any) -> Any:
    """
    Recursively convert a configuration object into a hashable representation.


    Handles various data types:
    - Converts dicts to sorted tuples of (key, value) pairs
    - Converts sets to sorted lists
    - Converts Path objects to strings
    - Preserves other basic types (str, int, float, bool, None)
    - Recursively processes nested structures (lists, tuples, dicts)

    Args:
        obj: Any configuration object that needs to be made hashable

    Returns:
        A hashable version of the input object
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(x) for x in obj)
    elif isinstance(obj, set):
        return tuple(sorted(_make_hashable(x) for x in obj))
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # For other objects, use their string representation
        return str(obj)


def generate_run_uuid(config: dict[str, Any]) -> str:
    """
    Generate a deterministic UUID v5 for a data processing configuration.

    Creates a unique identifier based on the contents of the configuration
    dictionary. The same configuration will always produce the same UUID,
    regardless of dict ordering, allowing for reproducible identification
    of processing runs.

    Args:
        config: Dictionary containing configuration parameters for data processing

    Returns:
        A UUID string uniquely identifying the configuration
    """
    # Convert the config to a hashable representation
    hashable_config = _make_hashable(config)

    # Serialize to a deterministic JSON string
    json_string = json.dumps(hashable_config, ensure_ascii=False)

    # Generate SHA256 hash of the serialized config
    config_bytes = json_string.encode("utf-8")
    sha256_hash = hashlib.sha256(config_bytes).hexdigest()

    # Create a UUID v5 using our namespace and the hash
    run_uuid = uuid.uuid5(PROCESSING_NAMESPACE, sha256_hash)

    return str(run_uuid)


def _default_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer to handle non-standard JSON types.

    Args:
        obj: Object to serialize to JSON

    Returns:
        JSON-serializable representation of the object

    Raises:
        TypeError: If the object cannot be serialized
    """
    if isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "isoformat"):
        # handles datetime, date, time, and numpy scalar types
        return obj.isoformat()
    elif hasattr(obj, 'steps') or (hasattr(obj, 'pipeline') and hasattr(obj.pipeline, 'steps')):
        # Handle pipeline objects (sklearn Pipeline, GroupedPipeline, UnifiedPipeline)
        return {
            'type': obj.__class__.__name__,
            'transformers': extract_pipeline_metadata(obj),
            'columns': getattr(obj, 'columns', None)
        }
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_config(config: dict[str, Any], filepath: Path) -> None:
    """
    Save a configuration dictionary to a JSON file.

    Args:
        config: Dictionary containing configuration parameters
        filepath: Path where the configuration will be saved

    Raises:
        FileOperationError: If there's an error creating directories or writing the file
        ConfigurationError: If there's an error serializing the configuration
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, sort_keys=True, default=_default_serializer)
    except (OSError, PermissionError) as e:
        raise FileOperationError(f"Failed to save configuration to {filepath}: {e}") from e
    except (TypeError, ValueError) as e:
        raise ConfigurationError(f"Failed to serialize configuration: {e}") from e
    except Exception as e:
        raise FileOperationError(f"Unexpected error saving configuration: {e}") from e


def load_config(filepath: Path) -> dict[str, Any]:
    """
    Load a configuration dictionary from a JSON file.

    Args:
        filepath: Path to the saved configuration file

    Returns:
        The loaded configuration dictionary

    Raises:
        FileOperationError: If the file doesn't exist or can't be read
        ConfigurationError: If the file content is invalid or not a dictionary
    """
    try:
        if not filepath.is_file():
            raise FileOperationError(f"Configuration file not found: {filepath}")
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
            raise ConfigurationError(f"Loaded object is not a dict, got {type(loaded_config)}")
        return loaded_config
    except (OSError, PermissionError) as e:
        raise FileOperationError(f"Failed to read configuration from {filepath}: {e}") from e
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file {filepath}: {e}") from e
    except (FileOperationError, ConfigurationError):
        # Re-raise our custom exceptions without wrapping
        raise
    except Exception as e:
        raise FileOperationError(f"Unexpected error loading configuration: {e}") from e


def extract_transformer_names(
    pipeline_obj: Union["Pipeline", "GroupedPipeline"],
) -> list[str]:
    """
    Extract the class names of transformers in a pipeline.

    Handles both sklearn Pipeline and GroupedPipeline objects.
    For GroupedPipeline, extracts names from the template pipeline.

    Args:
        pipeline_obj: A sklearn Pipeline or GroupedPipeline instance

    Returns:
        List of transformer class names in order of pipeline steps
    """
    from sklearn.pipeline import Pipeline

    # Handle GroupedPipeline
    if hasattr(pipeline_obj, "pipeline") and not isinstance(pipeline_obj, Pipeline):
        # For GroupedPipeline, use its pipeline attribute
        pipeline = pipeline_obj.pipeline
    else:
        # For sklearn Pipeline, use directly
        pipeline = pipeline_obj

    # Extract transformer class names from pipeline steps
    transformer_names = []
    for _, transformer in pipeline.steps:
        transformer_names.append(transformer.__class__.__name__)

    return transformer_names


def extract_pipeline_metadata(pipeline_obj: Any) -> list[str] | str:
    """
    Extract just the class names and order from a pipeline.
    
    This function extracts metadata from pipeline objects for serialization,
    replacing full object instances with their class name sequences.
    
    Args:
        pipeline_obj: A pipeline object (sklearn Pipeline, GroupedPipeline, or UnifiedPipeline)
        
    Returns:
        List of transformer class names for pipeline objects, or class name string for others
    """
    # Handle sklearn Pipeline
    if hasattr(pipeline_obj, 'steps'):
        return [step[1].__class__.__name__ for step in pipeline_obj.steps]
    # Handle GroupedPipeline or UnifiedPipeline with internal pipeline
    elif hasattr(pipeline_obj, 'pipeline') and hasattr(pipeline_obj.pipeline, 'steps'):
        return [step[1].__class__.__name__ for step in pipeline_obj.pipeline.steps]
    # For any other object, just return its class name
    return pipeline_obj.__class__.__name__


def extract_relevant_config(datamodule: "HydroLazyDataModule") -> dict[str, Any]:
    """
    Extract relevant configuration parameters from a HydroLazyDataModule instance.

    Creates a dictionary containing the key configuration parameters that define
    a data processing run. These parameters are used to generate a deterministic
    UUID for the run, ensuring reproducibility.

    Args:
        datamodule: An instance of HydroLazyDataModule

    Returns:
        Dictionary containing the relevant configuration parameters
    """
    # Extract relevant attributes that define the processing configuration
    relevant_config = {
        "forcing_features": datamodule.forcing_features,
        "static_features": datamodule.static_features,
        "target": datamodule.target,
        "group_identifier": datamodule.group_identifier,
        "is_autoregressive": datamodule.is_autoregressive,
        "train_prop": datamodule.train_prop,
        "val_prop": datamodule.val_prop,
        "test_prop": datamodule.test_prop,
        "min_train_years": datamodule.min_train_years,
        "max_imputation_gap_size": datamodule.max_imputation_gap_size,
        "gauge_ids_subset": datamodule.list_of_gauge_ids_to_process
        if datamodule.list_of_gauge_ids_to_process
        else None,
        "region_time_series_dirs": {k: str(v) for k, v in datamodule.region_time_series_base_dirs.items()},
        "region_static_attributes_dirs": {k: str(v) for k, v in datamodule.region_static_attributes_base_dirs.items()},
        "output_directory": str(datamodule.path_to_preprocessing_output_directory),
        "preprocessing_pipeline_keys": list(datamodule.preprocessing_configs.keys()),
    }

    # Add transformer details for each preprocessing pipeline
    transformer_details = {}
    for pipeline_key, pipeline_config in datamodule.preprocessing_configs.items():
        if "pipeline" in pipeline_config:
            pipeline_obj = pipeline_config["pipeline"]
            transformer_details[pipeline_key] = extract_transformer_names(pipeline_obj)

    # Add transformer details to config if any were found
    if transformer_details:
        relevant_config["preprocessing_transformer_details"] = transformer_details

    return relevant_config
