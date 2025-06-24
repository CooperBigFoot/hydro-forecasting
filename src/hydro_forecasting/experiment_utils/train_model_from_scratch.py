import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from ..exceptions import ConfigurationError, FileOperationError
from ..models import model_factory
from .training_runner import ExperimentRunner, ModelProviderFn

logger = logging.getLogger(__name__)


def create_new_model_from_hps(model_type: str, finalized_hps: dict[str, Any]) -> pl.LightningModule:
    """
    Create a new model from scratch using a finalized hyperparameter dictionary.
    """
    model, _ = model_factory.create_model_from_config_dict(model_type, finalized_hps)
    return model


def train_model_from_scratch(
    gauge_ids: list[str],
    datamodule_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: str,
    model_types: list[str],
    yaml_paths: list[str] | str,
    experiment_name: str,
    num_runs: int = 1,
    base_seed: int = 42,
    override_previous_attempts: bool = False,
) -> dict[str, tuple[str | None, dict[str, Any]]]:
    """
    Train one or more hydrological forecasting models from scratch.

    Returns:
        A dictionary mapping model types to tuples containing checkpoint paths and metrics.

    Raises:
        FileOperationError: If required YAML files don't exist.
        ConfigurationError: If model_types and yaml_paths lengths don't match.
    """
    actual_yaml_paths: list[str]
    if isinstance(yaml_paths, str):
        yaml_dir = Path(yaml_paths)
        if yaml_dir.is_dir():
            resolved_yaml_paths = []
            for model_type in model_types:
                yaml_path_obj = yaml_dir / f"{model_type.lower()}.yaml"
                if not yaml_path_obj.exists():
                    raise FileOperationError(f"YAML file for model type '{model_type}' not found at {yaml_path_obj}")
                resolved_yaml_paths.append(str(yaml_path_obj))
            actual_yaml_paths = resolved_yaml_paths
        else:
            logger.warning(
                "Provided yaml_paths is a string but not a directory. Using it as a single YAML path for all models."
            )
            actual_yaml_paths = [yaml_paths] * len(model_types)
    else:
        actual_yaml_paths = yaml_paths

    if len(model_types) != len(actual_yaml_paths):
        raise ConfigurationError("Length of model_types must match length of resolved yaml_paths")

    runner = ExperimentRunner(
        output_dir=output_dir,
        experiment_name=experiment_name,
        datamodule_config=datamodule_config,
        training_config=training_config,
        num_runs=num_runs,
        base_seed=base_seed,
        override_previous_attempts=override_previous_attempts,
    )

    model_provider_fns: list[ModelProviderFn] = [create_new_model_from_hps] * len(model_types)

    return runner.run_experiment(
        model_types=model_types,
        yaml_paths=actual_yaml_paths,
        model_provider_fns=model_provider_fns,
        gauge_ids=gauge_ids,
    )
