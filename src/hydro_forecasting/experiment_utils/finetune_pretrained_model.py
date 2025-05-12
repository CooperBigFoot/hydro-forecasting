import functools
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from returns.result import Failure, Result, Success

from ..experiment_utils import checkpoint_manager
from ..experiment_utils.training_runner import ExperimentRunner
from ..models import model_factory

logger = logging.getLogger(__name__)


def load_finetune_model(
    model_type: str,
    yaml_path: str,
    pretrained_checkpoint_path: str,
    lr_factor: float = 10.0,
) -> pl.LightningModule:
    """
    Load a pre-trained model for fine-tuning.

    Args:
        model_type: Type of model ('tide', 'tft', 'ealstm', etc.)
        yaml_path: Path to the YAML file with model hyperparameters
        pretrained_checkpoint_path: Path to the pre-trained checkpoint
        lr_factor: Factor by which to reduce the learning rate for fine-tuning

    Returns:
        Loaded model with reduced learning rate
    """
    model, _ = model_factory.load_pretrained_model(
        model_type=model_type,
        yaml_path=yaml_path,
        checkpoint_path=str(pretrained_checkpoint_path),
        lr_factor=lr_factor,
    )
    return model


def finetune_pretrained_models(
    # --- Pretrained Model Identification ---
    gauge_ids: list[str],
    pretrained_checkpoint_dir: str | Path,
    model_types: list[str],
    pretrained_yaml_paths: list[str | Path] | str | Path,
    datamodule_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: str | Path,
    experiment_name: str,
    select_best_from_pretrained: bool = True,
    pretrained_run_index: int | None = None,
    pretrained_attempt_index: int | None = None,
    lr_reduction_factor: float = 10.0,
    num_runs: int = 1,
    base_seed: int = 42,
    override_previous_attempts: bool = False,
) -> Result[dict[str, tuple[str, dict[str, Any]]], str]:
    """
    Fine-tune pre-trained hydrological forecasting models.

    This function orchestrates the fine-tuning process for one or more pre-trained models:
    1. Locates pre-trained checkpoints from a previous experiment
    2. Loads models with reduced learning rates
    3. Sets up data modules with the fine-tuning dataset
    4. Executes training loops for fine-tuning
    5. Manages versioned outputs and identifies best checkpoints
    6. Returns a summary of fine-tuning results

    Args:
        pretrained_checkpoint_dir: Directory containing pre-trained model checkpoints
        model_types: List of model types to fine-tune ('tide', 'tft', 'ealstm', etc.)
        pretrained_yaml_paths: List of paths to YAML files with model hyperparameters,
                              or a single path to a directory containing YAML files,
                              or a single path to a YAML file to be used for all model types
        select_best_from_pretrained: Whether to use the best checkpoint from pre-training
        pretrained_run_index: Specific run index to use when select_best_from_pretrained is False
        pretrained_attempt_index: Specific attempt index to use when select_best_from_pretrained is False
        lr_reduction_factor: Factor by which to reduce the learning rate for fine-tuning
        gauge_ids: List of basin/gauge IDs to use for fine-tuning
        datamodule_config: Configuration for the HydroInMemoryDataModule
        training_config: Configuration for PyTorch Lightning Trainer
        output_dir: Base directory for storing fine-tuning outputs
        experiment_name: Name of the fine-tuning experiment for directory organization
        num_runs: Number of independent fine-tuning runs for each model type
        base_seed: Base random seed (will be incremented for each run)
        override_previous_attempts: Whether to override existing fine-tuning outputs

    Returns:
        Result containing a dictionary mapping model types to tuples of
        (best_checkpoint_path, metrics_dict) if successful, or an error message if failed.
    """
    # Resolve pretrained_yaml_paths to a list if it's a string
    if isinstance(pretrained_yaml_paths, str | Path):
        yaml_dir = Path(pretrained_yaml_paths)
        if yaml_dir.is_dir():
            resolved_yaml_paths = []
            for model_type in model_types:
                yaml_path = yaml_dir / f"{model_type.lower()}.yaml"
                if not yaml_path.exists():
                    return Failure(f"YAML file for model type '{model_type}' not found at {yaml_path}")
                resolved_yaml_paths.append(str(yaml_path))
            pretrained_yaml_paths = resolved_yaml_paths
        else:
            logger.warning(
                "Provided pretrained_yaml_paths is a string but not a directory. Using it as a single YAML path for all models."
            )
            pretrained_yaml_paths = [pretrained_yaml_paths] * len(model_types)

    if len(model_types) != len(pretrained_yaml_paths):
        return Failure("Length of model_types must match length of pretrained_yaml_paths")

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

    pretrained_base_dir = Path(pretrained_checkpoint_dir)
    all_models_results = {}
    model_provider_fns = []

    # Locate pre-trained checkpoints for each model
    for model_type in model_types:
        pretrained_checkpoint_path_result = checkpoint_manager.get_checkpoint_path_to_load(
            base_checkpoint_load_dir=pretrained_base_dir,
            model_type=model_type,
            select_overall_best=select_best_from_pretrained,
            specific_run_index=pretrained_run_index,
            specific_attempt_index=pretrained_attempt_index,
        )

        if not isinstance(pretrained_checkpoint_path_result, Success):
            error_msg = pretrained_checkpoint_path_result.failure()
            logger.warning(f"Could not find pre-trained checkpoint for {model_type}: {error_msg}")
            all_models_results[model_type] = (None, {"error": f"Checkpoint not found: {error_msg}"})
            # Add a dummy provider function that will be skipped
            model_provider_fns.append(None)
        else:
            pretrained_checkpoint_path = pretrained_checkpoint_path_result.unwrap()
            logger.info(f"Found pre-trained checkpoint for {model_type}: {pretrained_checkpoint_path}")
            # Create a partially-applied function with the checkpoint path and lr factor
            model_provider_fn = functools.partial(
                load_finetune_model,
                pretrained_checkpoint_path=str(pretrained_checkpoint_path),
                lr_factor=lr_reduction_factor,
            )
            model_provider_fns.append(model_provider_fn)

    # Filter out models with missing checkpoints
    valid_model_types = []
    valid_yaml_paths = []
    valid_provider_fns = []

    for i, (model_type, yaml_path, provider_fn) in enumerate(
        zip(model_types, pretrained_yaml_paths, model_provider_fns, strict=True)
    ):
        if provider_fn is not None:
            valid_model_types.append(model_type)
            valid_yaml_paths.append(yaml_path)
            valid_provider_fns.append(provider_fn)
        else:
            # We already added an error entry to all_models_results for this model
            pass

    if not valid_model_types:
        return Failure("No models with valid pre-trained checkpoints found")

    # Run the experiment for valid models
    result = runner.run_experiment(
        model_types=valid_model_types,
        yaml_paths=valid_yaml_paths,
        model_provider_fns=valid_provider_fns,
        gauge_ids=gauge_ids,
    )

    # Merge results
    if isinstance(result, Success):
        all_models_results.update(result.unwrap())
        return Success(all_models_results)
    else:
        # If there was a failure with valid models, but we had some error entries,
        # still return those error entries
        if all_models_results:
            logger.warning(f"Experiment failed but returning partial results: {result.failure()}")
            return Success(all_models_results)
        else:
            return result
