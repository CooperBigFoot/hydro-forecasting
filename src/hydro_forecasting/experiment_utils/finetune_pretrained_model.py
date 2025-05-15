import functools
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from returns.result import Failure, Result, Success

from ..experiment_utils import checkpoint_manager
from ..models import model_factory
from .training_runner import ExperimentRunner, ModelProviderFn

logger = logging.getLogger(__name__)


def load_finetune_model_from_hps(
    model_type: str,
    finalized_hps: dict[str, Any],
    pretrained_checkpoint_path: str,
    lr_factor: float = 1.0,
) -> pl.LightningModule:
    """
    Load a pre-trained model for fine-tuning using finalized hyperparameters.
    The learning rate from finalized_hps will be adjusted by lr_factor.
    """

    model_config = model_factory.get_model_config_class(model_type)(**finalized_hps)
    model = model_factory.get_model_class(model_type).load_from_checkpoint(
        pretrained_checkpoint_path, config=model_config
    )

    if "learning_rate" in finalized_hps:
        original_lr = finalized_hps["learning_rate"]
        new_lr = original_lr / lr_factor
        if hasattr(model, "hparams") and "learning_rate" in model.hparams:
            model.hparams.learning_rate = new_lr
        elif hasattr(model, "learning_rate"):
            model.learning_rate = new_lr
        else:
            logger.warning(f"Could not set new learning rate on loaded model {model_type}")

        finalized_hps["original_lr_for_finetune"] = original_lr
        finalized_hps["learning_rate"] = new_lr
        logger.info(f"Finetuning {model_type}: Original LR {original_lr}, New LR {new_lr}")
    else:
        logger.warning(f"Cannot adjust LR for {model_type}: 'learning_rate' not in finalized_hps. Check YAML.")
    return model


def finetune_pretrained_models(
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
) -> Result[dict[str, tuple[str | None, dict[str, Any]]], str]:
    actual_pretrained_yaml_paths: list[str]
    if isinstance(pretrained_yaml_paths, str | Path):
        yaml_dir = Path(pretrained_yaml_paths)
        if yaml_dir.is_dir():
            resolved_yaml_paths = []
            for model_type in model_types:
                yaml_path_obj = yaml_dir / f"{model_type.lower()}.yaml"
                if not yaml_path_obj.exists():
                    return Failure(f"YAML file for model type '{model_type}' not found at {yaml_path_obj}")
                resolved_yaml_paths.append(str(yaml_path_obj))
            actual_pretrained_yaml_paths = resolved_yaml_paths
        else:
            logger.warning(
                "Provided pretrained_yaml_paths is a string but not a directory. Using it as a single YAML path for all models."
            )
            actual_pretrained_yaml_paths = [str(yaml_dir)] * len(model_types)
    else:
        actual_pretrained_yaml_paths = [str(p) for p in pretrained_yaml_paths]

    if len(model_types) != len(actual_pretrained_yaml_paths):
        return Failure("Length of model_types must match length of resolved pretrained_yaml_paths")

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
    model_provider_fns_for_exp: list[ModelProviderFn] = []
    valid_model_types_for_exp: list[str] = []
    valid_yaml_paths_for_exp: list[str] = []
    accumulated_error_results: dict[str, tuple[None, dict[str, Any]]] = {}

    for model_type, initial_yaml_path_str in zip(model_types, actual_pretrained_yaml_paths, strict=False):
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
            accumulated_error_results[model_type] = (None, {"error": f"Checkpoint not found: {error_msg}"})
        else:
            pretrained_checkpoint_path = pretrained_checkpoint_path_result.unwrap()
            logger.info(f"Found pre-trained checkpoint for {model_type}: {pretrained_checkpoint_path}")

            provider_fn = functools.partial(
                load_finetune_model_from_hps,
                pretrained_checkpoint_path=str(pretrained_checkpoint_path),
                lr_factor=lr_reduction_factor,
            )
            model_provider_fns_for_exp.append(provider_fn)
            valid_model_types_for_exp.append(model_type)
            valid_yaml_paths_for_exp.append(initial_yaml_path_str)

    if not valid_model_types_for_exp:
        if accumulated_error_results:
            return Success(accumulated_error_results)
        return Failure("No models with valid pre-trained checkpoints found and no prior errors.")

    experiment_run_result = runner.run_experiment(
        model_types=valid_model_types_for_exp,
        yaml_paths=valid_yaml_paths_for_exp,
        model_provider_fns=model_provider_fns_for_exp,
        gauge_ids=gauge_ids,
    )

    if isinstance(experiment_run_result, Success):
        final_results = experiment_run_result.unwrap()
        final_results.update(accumulated_error_results)
        return Success(final_results)
    else:
        if accumulated_error_results:
            logger.warning(
                f"ExperimentRunner failed but returning partial error results: {experiment_run_result.failure()}"
            )
            return Failure(
                f"ExperimentRunner failed: {experiment_run_result.failure()}. Prior errors: {accumulated_error_results}"
            )
        return experiment_run_result
