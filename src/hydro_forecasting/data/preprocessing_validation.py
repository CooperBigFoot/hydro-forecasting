from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError, PipelineCompatibilityError
from ..preprocessing.grouped import GroupedPipeline

if TYPE_CHECKING:
    pass


def validate_preprocessing_config_comprehensive(
    preprocessing_config: dict[str, dict[str, Any]],
    required_columns: list[str],
    group_identifier: str,
    available_columns: list[str] | None = None,
) -> None:
    """Main validation entry point - validates entire preprocessing config.

    Args:
        preprocessing_config: Dictionary mapping pipeline names to their configurations
        required_columns: List of columns that must be covered by pipelines
        group_identifier: The column used for grouping (e.g., 'gauge_id')
        available_columns: Optional list of all available columns for validation

    Raises:
        ConfigurationError: If configuration structure or values are invalid
        PipelineCompatibilityError: If pipeline compatibility issues are detected
    """
    if not preprocessing_config:
        raise ConfigurationError(
            "Preprocessing configuration is empty or None. "
            "At least one pipeline ('features' or 'target') must be configured."
        )

    # Validate individual pipeline configurations
    for pipeline_name, config in preprocessing_config.items():
        validate_pipeline_config(pipeline_name, config, group_identifier)

    # Validate strategy consistency across pipelines
    validate_strategy_consistency(preprocessing_config)

    # Validate column specifications if available columns are provided
    if available_columns:
        validate_column_specifications(preprocessing_config, available_columns, required_columns)


def validate_pipeline_config(pipeline_name: str, config: dict[str, Any], group_identifier: str) -> None:
    """Validate individual pipeline configuration.

    Args:
        pipeline_name: Name of the pipeline (e.g., 'features', 'target', 'static_features')
        config: Pipeline configuration dictionary
        group_identifier: The column used for grouping

    Raises:
        ConfigurationError: If configuration is invalid
        PipelineCompatibilityError: If pipeline object doesn't match strategy
    """
    # Schema validation - check required keys
    if not isinstance(config, dict):
        raise ConfigurationError(
            f"Pipeline '{pipeline_name}' configuration must be a dictionary. Got: {type(config).__name__}"
        )

    if "strategy" not in config:
        raise ConfigurationError(
            f"Pipeline '{pipeline_name}' missing required key 'strategy'. Current config: {config}"
        )

    if "pipeline" not in config:
        raise ConfigurationError(
            f"Pipeline '{pipeline_name}' missing required key 'pipeline'. Current config: {config}"
        )

    # Validate strategy value
    strategy = config["strategy"]
    valid_strategies = ["per_group", "unified"]
    if strategy not in valid_strategies:
        raise ConfigurationError(
            f"Invalid strategy '{strategy}' for pipeline '{pipeline_name}'. "
            f"Valid strategies are: {', '.join(repr(s) for s in valid_strategies)}. "
            f"Current config: {config}"
        )

    # Get pipeline object
    pipeline_obj = config["pipeline"]

    # Validate pipeline object
    validate_pipeline_object(pipeline_obj, strategy, pipeline_name)

    # Strategy-specific validation
    if strategy == "unified":
        validate_unified_strategy_config(pipeline_name, config)
    elif strategy == "per_group":
        validate_per_group_strategy_config(pipeline_name, config, pipeline_obj, group_identifier)


def validate_pipeline_object(pipeline_obj: Any, strategy: str, pipeline_name: str) -> None:
    """Validate the actual pipeline object matches requirements.

    Args:
        pipeline_obj: The pipeline object to validate
        strategy: The strategy ('per_group' or 'unified')
        pipeline_name: Name of the pipeline for error messages

    Raises:
        ConfigurationError: If pipeline object is invalid
        PipelineCompatibilityError: If pipeline doesn't match strategy
    """
    # Check for required methods
    required_methods = ["fit", "transform"]
    for method in required_methods:
        if not hasattr(pipeline_obj, method) or not callable(getattr(pipeline_obj, method)):
            raise ConfigurationError(
                f"Pipeline object for '{pipeline_name}' must have callable '{method}' method. "
                f"Got object of type: {type(pipeline_obj).__name__}"
            )

    # Check inverse_transform for target pipeline
    if pipeline_name == "target" and (
        not hasattr(pipeline_obj, "inverse_transform") or not callable(pipeline_obj.inverse_transform)
    ):
        raise ConfigurationError(
            f"Target pipeline must have callable 'inverse_transform' method for prediction postprocessing. "
            f"Got object of type: {type(pipeline_obj).__name__}"
        )

    # Validate pipeline type matches strategy
    if strategy == "per_group":
        if not isinstance(pipeline_obj, GroupedPipeline):
            raise PipelineCompatibilityError(
                f"Pipeline '{pipeline_name}' with strategy 'per_group' must be a GroupedPipeline instance. "
                f"Got: {type(pipeline_obj).__name__}"
            )
    elif strategy == "unified":
        # For unified strategy, accept sklearn Pipeline or BaseEstimator
        # UnifiedPipeline is created later by create_pipeline factory
        from sklearn.base import BaseEstimator
        from sklearn.pipeline import Pipeline

        if not isinstance(pipeline_obj, Pipeline | BaseEstimator):
            raise PipelineCompatibilityError(
                f"Pipeline '{pipeline_name}' with strategy 'unified' must be an sklearn Pipeline or BaseEstimator. "
                f"Got: {type(pipeline_obj).__name__}"
            )


def validate_unified_strategy_config(pipeline_name: str, config: dict[str, Any]) -> None:
    """Validate configuration specific to unified strategy.

    Args:
        pipeline_name: Name of the pipeline
        config: Pipeline configuration dictionary

    Raises:
        ConfigurationError: If unified strategy requirements are not met
    """
    # Unified strategy must have columns specification
    if "columns" not in config:
        raise ConfigurationError(
            f"Pipeline '{pipeline_name}' with unified strategy must specify 'columns'. Current config: {config}"
        )

    columns = config["columns"]
    if not isinstance(columns, list):
        raise ConfigurationError(
            f"Pipeline '{pipeline_name}' columns must be a list of strings. Got: {type(columns).__name__}"
        )

    if not columns:
        raise ConfigurationError(f"Pipeline '{pipeline_name}' columns list cannot be empty. Current config: {config}")

    # Validate all column names are strings
    for col in columns:
        if not isinstance(col, str):
            raise ConfigurationError(
                f"Pipeline '{pipeline_name}' column names must be strings. "
                f"Found non-string column: {col} (type: {type(col).__name__})"
            )

    # Validate fit_on_n_basins if present
    if "fit_on_n_basins" in config:
        fit_on_n_basins = config["fit_on_n_basins"]
        if not isinstance(fit_on_n_basins, int) or fit_on_n_basins <= 0:
            raise ConfigurationError(
                f"Pipeline '{pipeline_name}' fit_on_n_basins must be a positive integer. Got: {fit_on_n_basins}"
            )


def validate_per_group_strategy_config(
    pipeline_name: str, config: dict[str, Any], pipeline_obj: GroupedPipeline, group_identifier: str
) -> None:
    """Validate configuration specific to per_group strategy.

    Args:
        pipeline_name: Name of the pipeline
        config: Pipeline configuration dictionary
        pipeline_obj: The GroupedPipeline instance
        group_identifier: Expected group identifier

    Raises:
        ConfigurationError: If per_group strategy requirements are not met
    """
    # Validate GroupedPipeline has correct group_identifier
    if not hasattr(pipeline_obj, "group_identifier"):
        raise ConfigurationError(f"GroupedPipeline for '{pipeline_name}' must have 'group_identifier' attribute")

    if pipeline_obj.group_identifier != group_identifier:
        raise ConfigurationError(
            f"GroupedPipeline for '{pipeline_name}' has group_identifier '{pipeline_obj.group_identifier}' "
            f"but expected '{group_identifier}'"
        )

    # Per_group strategy should not have fit_on_n_basins
    if "fit_on_n_basins" in config:
        raise ConfigurationError(
            f"Pipeline '{pipeline_name}' with per_group strategy should not have 'fit_on_n_basins'. "
            f"This parameter is only valid for unified strategy."
        )


def validate_strategy_consistency(preprocessing_config: dict[str, dict[str, Any]]) -> None:
    """Ensure strategy choices are compatible across pipelines.

    Args:
        preprocessing_config: Complete preprocessing configuration

    Raises:
        ConfigurationError: If strategies are incompatible
    """
    # Extract strategies
    strategies = {}
    for pipeline_name, config in preprocessing_config.items():
        strategies[pipeline_name] = config.get("strategy")

    # Check if mixing strategies (this is actually allowed, just log it)
    unique_strategies = set(strategies.values())
    if len(unique_strategies) > 1:
        # This is allowed but worth noting
        pass


def validate_column_specifications(
    preprocessing_config: dict[str, dict[str, Any]], available_columns: list[str], required_columns: list[str]
) -> None:
    """Validate column specifications when available_columns is known.

    Args:
        preprocessing_config: Complete preprocessing configuration
        available_columns: List of all available columns in the data
        required_columns: List of columns that must be covered

    Raises:
        ConfigurationError: If column specifications are invalid
    """
    # Collect all specified columns across pipelines
    specified_columns: dict[str, list[str]] = {}

    for pipeline_name, config in preprocessing_config.items():
        if "columns" in config:
            columns = config["columns"]
            specified_columns[pipeline_name] = columns

            # Check all specified columns exist in available columns
            for col in columns:
                if col not in available_columns:
                    raise ConfigurationError(
                        f"Pipeline '{pipeline_name}' specifies column '{col}' "
                        f"which is not in available columns. "
                        f"Available columns: {sorted(available_columns)}"
                    )

    # Special validation for features and target
    if "features" in specified_columns and "target" in specified_columns:
        features_cols = set(specified_columns["features"])
        target_cols = set(specified_columns["target"])

        # Target columns should not be in features (except group_identifier)
        overlap = features_cols.intersection(target_cols)
        if overlap:
            # Group identifier is allowed to be in both
            overlap_without_group = overlap - {available_columns[0]} if available_columns else overlap
            if overlap_without_group:
                raise ConfigurationError(
                    f"Target columns {sorted(overlap_without_group)} should not be included "
                    f"in features pipeline columns. This can cause data leakage."
                )

    # Check all required columns are covered
    all_specified = set()
    for cols in specified_columns.values():
        all_specified.update(cols)

    # For per_group strategy without explicit columns, assume it processes all columns
    for config in preprocessing_config.values():
        if config.get("strategy") == "per_group" and "columns" not in config:
            # Per-group pipelines without explicit columns process all columns
            all_specified.update(required_columns)

    missing_required = set(required_columns) - all_specified
    if missing_required:
        raise ConfigurationError(
            f"Required columns {sorted(missing_required)} are not covered by any pipeline. "
            f"Ensure all required columns are specified in at least one pipeline configuration."
        )


def validate_preprocessing_pipelines(
    features_pipeline: Any | None,
    target_pipeline: Any | None,
    preprocessing_config: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Validate preprocessing pipeline objects for compatibility.

    This is a compatibility function that validates pipeline objects directly.

    Args:
        features_pipeline: Feature preprocessing pipeline
        target_pipeline: Target preprocessing pipeline
        preprocessing_config: Optional preprocessing configuration

    Raises:
        ConfigurationError: If pipelines are invalid
    """
    # At least one pipeline must be provided
    if features_pipeline is None and target_pipeline is None:
        raise ConfigurationError("At least one preprocessing pipeline (features or target) must be provided")

    # Validate individual pipelines if provided
    if features_pipeline is not None:
        required_methods = ["fit", "transform"]
        for method in required_methods:
            if not hasattr(features_pipeline, method) or not callable(getattr(features_pipeline, method)):
                raise ConfigurationError(f"Features pipeline must have callable '{method}' method")

    if target_pipeline is not None:
        required_methods = ["fit", "transform", "inverse_transform"]
        for method in required_methods:
            if not hasattr(target_pipeline, method) or not callable(getattr(target_pipeline, method)):
                raise ConfigurationError(f"Target pipeline must have callable '{method}' method")
