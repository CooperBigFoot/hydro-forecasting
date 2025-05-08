import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from returns.result import Failure, Result, Success
from sklearn.pipeline import Pipeline

from ..preprocessing.grouped import GroupedPipeline

if TYPE_CHECKING:
    from .in_memory_datamodule import HydroInMemoryDataModule


def validate_positive_integer(param_name: str, value: Any) -> Result[None, str]:
    """Validate that a parameter is a positive integer."""
    if not isinstance(value, int):
        return Failure(f"Parameter '{param_name}' must be an integer, got {type(value).__name__}")
    if value <= 0:
        return Failure(f"Parameter '{param_name}' must be greater than 0, got {value}")
    return Success(None)


def validate_non_negative_integer(param_name: str, value: Any) -> Result[None, str]:
    """Validate that a parameter is a non-negative integer."""
    if not isinstance(value, int):
        return Failure(f"Parameter '{param_name}' must be an integer, got {type(value).__name__}")
    if value < 0:
        return Failure(f"Parameter '{param_name}' must be greater than or equal to 0, got {value}")
    return Success(None)


def validate_positive_float(param_name: str, value: Any) -> Result[None, str]:
    """Validate that a parameter is a positive float."""
    if not isinstance(value, (int, float)):
        return Failure(f"Parameter '{param_name}' must be a number, got {type(value).__name__}")
    if value <= 0:
        return Failure(f"Parameter '{param_name}' must be greater than 0, got {value}")
    return Success(None)


def validate_string_list(param_name: str, value: Any, allow_empty: bool = False) -> Result[None, str]:
    """Validate that a parameter is a list of strings."""
    if not isinstance(value, list):
        return Failure(f"Parameter '{param_name}' must be a list, got {type(value).__name__}")
    if not allow_empty and not value:
        return Failure(f"Parameter '{param_name}' must not be an empty list.")
    if not all(isinstance(item, str) for item in value):
        return Failure(f"All items in '{param_name}' must be strings.")
    return Success(None)


def validate_non_empty_string(param_name: str, value: Any) -> Result[None, str]:
    """Validate that a parameter is a non-empty string."""
    if not isinstance(value, str):
        return Failure(f"Parameter '{param_name}' must be a string, got {type(value).__name__}")
    if not value:
        return Failure(f"Parameter '{param_name}' must not be an empty string.")
    return Success(None)


def validate_boolean(param_name: str, value: Any) -> Result[None, str]:
    """Validate that a parameter is a boolean."""
    if not isinstance(value, bool):
        return Failure(f"Parameter '{param_name}' must be a boolean, got {type(value).__name__}")
    return Success(None)


def validate_path_dict(param_name: str, value: Any, check_existence: bool = False) -> Result[None, str]:
    """Validate that a parameter is a dictionary of string keys and Path values."""
    if not isinstance(value, dict):
        return Failure(f"Parameter '{param_name}' must be a dictionary, got {type(value).__name__}")
    for k, v in value.items():
        if not isinstance(k, str):
            return Failure(f"Keys in '{param_name}' must be strings, got {type(k).__name__}")
        if not isinstance(v, (str, Path)):
            return Failure(
                f"Values in '{param_name}' must be strings or Path objects, got {type(v).__name__} for key '{k}'"
            )
        path_v = Path(v)
        if check_existence and not path_v.exists():
            return Failure(f"Path '{path_v}' for key '{k}' in '{param_name}' does not exist.")
        if check_existence and not path_v.is_dir():
            return Failure(f"Path '{path_v}' for key '{k}' in '{param_name}' is not a directory.")
    return Success(None)


def validate_path(
    param_name: str,
    value: Any,
    must_exist: bool = False,
    must_be_dir: bool = False,
) -> Result[None, str]:
    """Validate that a parameter is a Path object and optionally check existence/type."""
    if not isinstance(value, (str, Path)):
        return Failure(f"Parameter '{param_name}' must be a string or Path object, got {type(value).__name__}")
    path_value = Path(value)
    if must_exist and not path_value.exists():
        return Failure(f"Path for '{param_name}': '{path_value}' does not exist.")
    if must_exist and must_be_dir and not path_value.is_dir():
        return Failure(f"Path for '{param_name}': '{path_value}' is not a directory.")
    return Success(None)


def validate_train_val_test_proportions(train_p: float, val_p: float, test_p: float) -> Result[None, str]:
    """Validate that train, val, and test proportions sum to approximately 1."""
    if not all(isinstance(p, (int, float)) and 0.0 <= p <= 1.0 for p in [train_p, val_p, test_p]):
        return Failure("Train, val, and test proportions must be floats between 0.0 and 1.0.")
    total_prop = math.fsum([train_p, val_p, test_p])
    if not math.isclose(total_prop, 1.0, abs_tol=1e-6):
        return Failure(f"Training, validation, and test proportions must sum to 1.0. Current sum: {total_prop}")
    return Success(None)


def validate_target_in_features(
    target: str,
    forcing_features: list[str],
    # is_autoregressive argument is removed
) -> Result[None, str]:
    """
    Validate that the target variable is NOT included in the user-provided forcing_features list.

    Autoregression (using past target as input) should be controlled solely by the
    is_autoregressive flag in the DataModule/Dataset, not by adding the target to forcing_features.
    """
    if target in forcing_features:
        return Failure(
            f"Target variable '{target}' should NOT be included in the 'forcing_features' list. \n"
            f"To use the past target as an input feature (autoregression), "
            f"set the 'is_autoregressive' flag to True in the DataModule configuration. \n"
            f"The DataModule/Dataset handles adding the target internally when this flag is set."
        )
    return Success(None)


def _validate_pipeline_compatibility(
    pipeline_obj: Union[Pipeline, GroupedPipeline], pipeline_name: str
) -> Result[None, str]:
    """Helper to validate steps of a single pipeline."""
    if isinstance(pipeline_obj, GroupedPipeline):
        # Validate the template pipeline within GroupedPipeline
        if not hasattr(pipeline_obj, "pipeline") or not isinstance(pipeline_obj.pipeline, Pipeline):
            return Failure(
                f"GroupedPipeline '{pipeline_name}' does not have a valid inner sklearn.pipeline.Pipeline attribute."
            )
        p_to_check = pipeline_obj.pipeline
    elif isinstance(pipeline_obj, Pipeline):
        p_to_check = pipeline_obj
    else:
        return Failure(f"'{pipeline_name}' is not a valid Pipeline or GroupedPipeline instance.")

    for step_name, transformer in p_to_check.steps:
        if not (hasattr(transformer, "fit") and callable(getattr(transformer, "fit"))):
            return Failure(f"Transformer '{step_name}' in pipeline '{pipeline_name}' is missing a 'fit' method.")
        if not (hasattr(transformer, "transform") and callable(getattr(transformer, "transform"))):
            return Failure(f"Transformer '{step_name}' in pipeline '{pipeline_name}' is missing a 'transform' method.")
        if not (hasattr(transformer, "inverse_transform") and callable(getattr(transformer, "inverse_transform"))):
            return Failure(
                f"Transformer '{step_name}' in pipeline '{pipeline_name}' is missing an 'inverse_transform' method."
            )
    return Success(None)


def validate_preprocessing_pipelines_config(
    preprocessing_configs_attr: Optional[dict[str, dict[str, Any]]],
    datamodule_group_identifier: str,
) -> Result[None, str]:
    """Validate the structure and compatibility of preprocessing_configs."""
    if preprocessing_configs_attr is None:
        return Success(None)  # No preprocessing configs to validate
    if not isinstance(preprocessing_configs_attr, dict):
        return Failure("'preprocessing_configs' must be a dictionary.")

    for pipeline_type, config_dict in preprocessing_configs_attr.items():
        if not isinstance(config_dict, dict):
            return Failure(f"Config for pipeline type '{pipeline_type}' must be a dictionary.")
        if "pipeline" not in config_dict:
            return Failure(f"Missing 'pipeline' key in '{pipeline_type}' config.")

        pipeline_obj = config_dict["pipeline"]

        # Validate pipeline compatibility (fit, transform, inverse_transform methods)
        comp_result = _validate_pipeline_compatibility(pipeline_obj, pipeline_type)
        if isinstance(comp_result, Failure):
            return comp_result

        # Specific checks for GroupedPipeline
        if isinstance(pipeline_obj, GroupedPipeline):
            if not hasattr(pipeline_obj, "group_identifier"):
                return Failure(f"GroupedPipeline for '{pipeline_type}' is missing 'group_identifier' attribute.")
            if pipeline_obj.group_identifier != datamodule_group_identifier:
                return Failure(
                    f"GroupedPipeline for '{pipeline_type}' has group_identifier '{pipeline_obj.group_identifier}', "
                    f"but DataModule uses '{datamodule_group_identifier}'."
                )

        # Specific checks for 'static_features'
        if pipeline_type == "static_features":
            if "columns" not in config_dict:
                return Failure("Config for 'static_features' pipeline must include a 'columns' key.")
            if not isinstance(config_dict["columns"], list) or not all(
                isinstance(c, str) for c in config_dict["columns"]
            ):
                return Failure("'columns' for 'static_features' must be a list of strings.")
    return Success(None)


def validate_hydro_inmemory_datamodule_config(
    dm: "HydroInMemoryDataModule",
) -> Result[None, str]:
    """Orchestrates all validation checks for the HydroInMemoryDataModule."""
    return (
        Success(None)
        .bind(
            lambda _: validate_path_dict(
                "region_time_series_base_dirs",
                dm.hparams.region_time_series_base_dirs,
                check_existence=False,
            )
        )  # Existence checked during loading
        .bind(
            lambda _: validate_path_dict(
                "region_static_attributes_base_dirs",
                dm.hparams.region_static_attributes_base_dirs,
                check_existence=False,
            )
        )  # Existence checked during loading
        .bind(
            lambda _: validate_path(
                "path_to_preprocessing_output_directory",
                dm.hparams.path_to_preprocessing_output_directory,
                must_exist=False,
            )
        )  # Will be created
        .bind(lambda _: validate_non_empty_string("group_identifier", dm.hparams.group_identifier))
        .bind(lambda _: validate_positive_integer("batch_size", dm.hparams.batch_size))
        .bind(lambda _: validate_positive_integer("input_length", dm.hparams.input_length))
        .bind(lambda _: validate_positive_integer("output_length", dm.hparams.output_length))
        .bind(
            lambda _: validate_string_list(
                "forcing_features",
                dm.hparams.forcing_features,
                allow_empty=False,
            )
        )
        .bind(lambda _: validate_string_list("static_features", dm.hparams.static_features, allow_empty=True))
        .bind(lambda _: validate_non_empty_string("target", dm.hparams.target))
        .bind(lambda _: validate_preprocessing_pipelines_config(dm.preprocessing_configs, dm.hparams.group_identifier))
        .bind(lambda _: validate_non_negative_integer("num_workers", dm.hparams.num_workers))  # num_workers can be 0
        .bind(lambda _: validate_positive_float("min_train_years", dm.hparams.min_train_years))
        .bind(
            lambda _: validate_train_val_test_proportions(
                dm.hparams.train_prop, dm.hparams.val_prop, dm.hparams.test_prop
            )
        )
        .bind(lambda _: validate_non_negative_integer("max_imputation_gap_size", dm.hparams.max_imputation_gap_size))
        .bind(lambda _: validate_positive_integer("chunk_size", dm.hparams.chunk_size))
        .bind(
            lambda _: validate_string_list(
                "list_of_gauge_ids_to_process",
                dm.hparams.list_of_gauge_ids_to_process if dm.hparams.list_of_gauge_ids_to_process is not None else [],
                allow_empty=True,
            )
        )
        .bind(lambda _: validate_non_empty_string("domain_id", dm.hparams.domain_id))
        .bind(lambda _: validate_non_empty_string("domain_type", dm.hparams.domain_type))
        .bind(lambda _: validate_boolean("is_autoregressive", dm.hparams.is_autoregressive))
        .bind(lambda _: validate_target_in_features(dm.hparams.target, dm.hparams.forcing_features))
    )
