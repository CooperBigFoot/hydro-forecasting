import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from returns.result import Result

from ..experiment_utils.training_runner import ExperimentRunner
from ..models import model_factory

logger = logging.getLogger(__name__)


def create_new_model(model_type: str, yaml_path: str) -> pl.LightningModule:
    """
    Create a new model from scratch.

    Args:
        model_type: Type of model ('tide', 'tft', 'ealstm', etc.)
        yaml_path: Path to the YAML file with model hyperparameters

    Returns:
        Initialized model
    """
    model, _ = model_factory.create_model(model_type, yaml_path)
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
) -> Result[dict[str, tuple[str, dict[str, float]]], str]:
    """
    Train one or more hydrological forecasting models from scratch.

    This function serves as a high-level orchestrator for training deep learning models
    within the hydrological forecasting framework. It manages the entire process from
    data preparation to model training and checkpoint management.

    Args:
        gauge_ids: List of basin/gauge IDs to use for training
        datamodule_config: Configuration for the HydroInMemoryDataModule
        training_config: Configuration for PyTorch Lightning Trainer
        output_dir: Base directory for storing outputs (checkpoints, logs)
        model_types: List of model types to train ('tide', 'tft', 'ealstm', etc.)
        yaml_paths: List of paths to YAML files with model hyperparameters,
                   or a single path to a directory containing YAML files named after model types
                   (e.g., '/path/to/yamls/tide.yaml' for model_type 'tide'),
                   or a single path to a YAML file to be used for all model types
        experiment_name: Name of the experiment for directory organization
        num_runs: Number of independent training runs for each model type
        base_seed: Base random seed (will be incremented for each run)
        override_previous_attempts: Whether to override existing outputs

    Returns:
        Result containing a dictionary mapping model types to tuples of
        (best_checkpoint_path, metrics_dict) if successful, or an error message if failed.
    """
    # Resolve yaml_paths to a list if it's a string
    if isinstance(yaml_paths, str):
        yaml_dir = Path(yaml_paths)
        if yaml_dir.is_dir():
            resolved_yaml_paths = []
            for model_type in model_types:
                yaml_path = yaml_dir / f"{model_type.lower()}.yaml"
                if not yaml_path.exists():
                    return Result.from_failure(f"YAML file for model type '{model_type}' not found at {yaml_path}")
                resolved_yaml_paths.append(str(yaml_path))
            yaml_paths = resolved_yaml_paths
        else:
            logger.warning(
                "Provided yaml_paths is a string but not a directory. Using it as a single YAML path for all models."
            )
            yaml_paths = [yaml_paths] * len(model_types)

    if len(model_types) != len(yaml_paths):
        return Result.from_failure("Length of model_types must match length of yaml_paths")

    # Create experiment runner
    runner = ExperimentRunner(
        output_dir=output_dir,
        experiment_name=experiment_name,
        datamodule_config=datamodule_config,
        training_config=training_config,
        num_runs=num_runs,
        base_seed=base_seed,
        override_previous_attempts=override_previous_attempts,
    )

    # Create list of model provider functions (all using create_new_model)
    model_provider_fns = [create_new_model] * len(model_types)

    # Run the experiment
    return runner.run_experiment(
        model_types=model_types,
        yaml_paths=yaml_paths,
        model_provider_fns=model_provider_fns,
        gauge_ids=gauge_ids,
    )
