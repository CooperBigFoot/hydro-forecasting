import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Union, TypeVar
import joblib

from sklearn.pipeline import Pipeline
from ..preprocessing.grouped import GroupedPipeline

# Fixed namespace for hydro-forecasting processing runs
PROCESSING_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")

# Type variable for forward reference to HydroLazyDataModule
T = TypeVar('T')

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
        
    Example:
        >>> config = {
        ...     "input_length": 365,
        ...     "output_length": 30,
        ...     "forcing_features": ["precipitation", "temperature"],
        ...     "paths": {"data": "/path/to/data"}
        ... }
        >>> generate_run_uuid(config)
        '6c45cbcd-e80d-5da5-a7ca-a3c9dbeb2a84'
    """
    # Convert the config to a hashable representation
    hashable_config = _make_hashable(config)
    
    # Serialize to a deterministic JSON string
    json_string = json.dumps(hashable_config, ensure_ascii=False)
    
    # Generate SHA256 hash of the serialized config
    config_bytes = json_string.encode('utf-8')
    sha256_hash = hashlib.sha256(config_bytes).hexdigest()
    
    # Create a UUID v5 using our namespace and the hash
    run_uuid = uuid.uuid5(PROCESSING_NAMESPACE, sha256_hash)
    
    return str(run_uuid)

def save_pipelines(
    pipelines: dict[str, Union[Pipeline, GroupedPipeline]], 
    filepath: Path
) -> None:
    """
    Save a dictionary of scikit-learn Pipeline or GroupedPipeline objects to disk.
    
    Uses joblib for efficient serialization of fitted pipelines, with error
    handling via the Result monad.
    
    Args:
        pipelines: Dictionary mapping identifiers to Pipeline or GroupedPipeline objects
        filepath: Path where the pipelines will be saved
        
    Returns:
        Success with None if saving was successful, or Failure with error message
        
    Example:
        >>> pipelines = {"basin_1": fitted_pipeline, "basin_2": another_pipeline}
        >>> result = save_pipelines(pipelines, Path("models/basin_pipelines.joblib"))
        >>> result.map(lambda _: print("Pipelines saved successfully"))
    """
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Save pipelines using joblib (will raise on failure)
    joblib.dump(pipelines, filepath)

def load_pipelines(
    filepath: Path
) -> dict[str, Union[Pipeline, GroupedPipeline]]:
    """
    Load a dictionary of scikit-learn Pipeline or GroupedPipeline objects from disk.
    
    Retrieves previously saved pipelines with error handling via the Result monad.
    
    Args:
        filepath: Path to the saved pipeline file
        
    Returns:
        Success with the loaded pipeline dictionary if successful, 
        or Failure with error message
        
    Example:
        >>> result = load_pipelines(Path("models/basin_pipelines.joblib"))
        >>> result.map(
        ...     lambda pipelines: print(f"Loaded {len(pipelines)} pipelines")
        ... ).alt(
        ...     lambda error: print(f"Error: {error}")
        ... )
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Pipeline file not found: {filepath}")
    loaded_pipelines = joblib.load(filepath)
    if not isinstance(loaded_pipelines, dict):
        raise TypeError(f"Loaded object is not a dict, got {type(loaded_pipelines)}")
    return loaded_pipelines

def _default_serializer(obj: Any) -> str:
    """
    Custom JSON serializer to handle non-standard JSON types.
    
    Args:
        obj: Object to serialize to JSON
        
    Returns:
        String representation of the object
        
    Raises:
        TypeError: If the object cannot be serialized
    """
    if isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "isoformat"):
        # handles datetime, date, time, and numpy scalar types
        return obj.isoformat()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_config(config: dict[str, Any], filepath: Path) -> None:
    """
    Save a configuration dictionary to a JSON file.
    
    Uses the Result monad for error handling to provide a functional approach
    to dealing with IO operations that might fail.
    
    Args:
        config: Dictionary containing configuration parameters
        filepath: Path where the configuration will be saved
        
    Returns:
        Success with None if saving was successful, or Failure with error message
        
    Example:
        >>> config = {"input_length": 365, "batch_size": 32}
        >>> result = save_config(config, Path("configs/model_config.json"))
        >>> result.map(lambda _: print("Config saved successfully"))
    """ 
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Save the config as JSON (will raise on failure)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, sort_keys=True, default=_default_serializer)

def load_config(filepath: Path) -> dict[str, Any]:
    """
    Load a configuration dictionary from a JSON file.
    
    Uses the Result monad for error handling to provide a functional approach
    to dealing with IO operations that might fail.
    
    Args:
        filepath: Path to the saved configuration file
        
    Returns:
        Success with the loaded configuration dictionary if successful,
        or Failure with error message
        
    Example:
        >>> result = load_config(Path("configs/model_config.json"))
        >>> result.map(
        ...     lambda config: print(f"Loaded config with {len(config)} parameters")
        ... ).alt(
        ...     lambda error: print(f"Error: {error}")
        ... )
    """ 
    if not filepath.is_file():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
    if not isinstance(loaded_config, dict):
        raise TypeError(f"Loaded object is not a dict, got {type(loaded_config)}")
    return loaded_config

def extract_relevant_config(datamodule: T) -> dict[str, Any]:
    """
    Extract relevant configuration parameters from a HydroLazyDataModule instance.
    
    Creates a dictionary containing the key configuration parameters that define
    a data processing run. These parameters are used to generate a deterministic
    UUID for the run, ensuring reproducibility.
    
    Args:
        datamodule: An instance of HydroLazyDataModule
        
    Returns:
        Dictionary containing the relevant configuration parameters
        
    Example:
        >>> config = extract_relevant_config(datamodule_instance)
        >>> uuid = generate_run_uuid(config)
        >>> print(f"Processing run UUID: {uuid}")
    """ 
    # Extract relevant attributes that define the processing configuration
    relevant_config = {
        # Input/output parameters
        "input_length": datamodule.input_length,
        "output_length": datamodule.output_length,
        # Features and target
        "forcing_features": datamodule.forcing_features,
        "static_features": datamodule.static_features,
        "target": datamodule.target,
        # Dataset properties
        "group_identifier": datamodule.group_identifier,
        "is_autoregressive": datamodule.is_autoregressive,
        "domain_id": datamodule.domain_id,
        "domain_type": datamodule.domain_type,
        # Data splitting parameters
        "train_prop": datamodule.train_prop,
        "val_prop": datamodule.val_prop,
        "test_prop": datamodule.test_prop,
        "min_train_years": datamodule.min_train_years,
        # Processing parameters
        "max_imputation_gap_size": datamodule.max_imputation_gap_size,
        # Only include gauge IDs if they were explicitly provided
        "gauge_ids_subset": datamodule.list_of_gauge_ids_to_process if datamodule.list_of_gauge_ids_to_process else None,
        # Resource paths
        "region_time_series_dirs": {k: str(v) for k, v in datamodule.region_time_series_base_dirs.items()},
        "region_static_attributes_dirs": {k: str(v) for k, v in datamodule.region_static_attributes_base_dirs.items()},
        "output_directory": str(datamodule.path_to_preprocessing_output_directory),
        # Preprocessing pipeline configuration keys
        # We don't include actual pipeline objects as they can't be JSON serialized
        # Instead, we store the keys which identify the pipeline configurations
        "preprocessing_pipeline_keys": list(datamodule.preprocessing_configs.keys()),
    }
    
    return relevant_config
