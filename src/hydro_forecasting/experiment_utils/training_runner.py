import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from returns.result import Failure, Result, Success

from ..data.in_memory_datamodule import HydroInMemoryDataModule
from ..experiment_utils import checkpoint_manager
from ..model_evaluation.hp_from_yaml import hp_from_yaml

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=pl.LightningModule)

ModelProviderFn = Callable[[str, dict[str, Any]], ModelType]

torch.set_float32_matmul_precision("medium")


def _setup_datamodule_core(
    base_datamodule_config: dict[str, Any],
    hps_for_datamodule: dict[str, Any],
    gauge_ids: list[str],
    model_type: str,  # Added for clarity in case of missing HPs
) -> Result[HydroInMemoryDataModule, str]:
    """
    Core helper to configure and set up HydroInMemoryDataModule.
    """
    current_datamodule_config = copy.deepcopy(base_datamodule_config)

    if "input_length" in hps_for_datamodule:
        current_datamodule_config["input_length"] = hps_for_datamodule["input_length"]
    elif "input_len" in hps_for_datamodule:
        current_datamodule_config["input_length"] = hps_for_datamodule["input_len"]
    else:
        if "input_len" not in hps_for_datamodule and "input_length" not in hps_for_datamodule:
            logger.warning(
                f"Neither 'input_length' nor 'input_len' found in hps_for_datamodule for {model_type}. "
                f"Using value from base_datamodule_config: {current_datamodule_config.get('input_length')}"
            )

    if "output_length" in hps_for_datamodule:
        current_datamodule_config["output_length"] = hps_for_datamodule["output_length"]
    elif "output_len" in hps_for_datamodule:
        current_datamodule_config["output_length"] = hps_for_datamodule["output_len"]
    else:
        if "output_len" not in hps_for_datamodule and "output_length" not in hps_for_datamodule:
            logger.warning(
                f"Neither 'output_length' nor 'output_len' found in hps_for_datamodule for {model_type}. "
                f"Using value from base_datamodule_config: {current_datamodule_config.get('output_length')}"
            )

    if "batch_size" in hps_for_datamodule:
        current_datamodule_config["batch_size"] = hps_for_datamodule["batch_size"]

    try:
        datamodule = HydroInMemoryDataModule(list_of_gauge_ids_to_process=gauge_ids, **current_datamodule_config)
        datamodule.prepare_data()
        datamodule.setup()
        return Success(datamodule)
    except Exception as e:
        logger.error(f"Failed to setup datamodule for {model_type}: {e}", exc_info=True)
        return Failure(f"Datamodule setup failed for {model_type}: {e}")


def _finalize_model_hyperparameters(
    model_hps: dict[str, Any],
    datamodule: HydroInMemoryDataModule,
    model_type: str,  # Added for logging context
) -> dict[str, Any]:
    """
    Enriches model HPs with datamodule-derived dimensions and other common mappings.
    """
    final_hps = copy.deepcopy(model_hps)

    # Map common aliases if present
    if "input_length" in final_hps and "input_len" not in final_hps:
        final_hps["input_len"] = final_hps.pop("input_length")
    if "output_length" in final_hps and "output_len" not in final_hps:
        final_hps["output_len"] = final_hps.pop("output_length")

    # Add/overwrite with datamodule-derived properties
    try:
        final_hps["input_size"] = len(datamodule.input_features_ordered_for_X)
        final_hps["static_size"] = len(datamodule.static_features) if datamodule.static_features else 0
        final_hps["future_input_size"] = len(datamodule.future_features_ordered)
        final_hps["group_identifier"] = datamodule.group_identifier
        # Ensure output_len in HPs matches datamodule's output_length after setup
        final_hps["output_len"] = datamodule.output_length
    except Exception as e:
        logger.error(f"Error finalizing HPs for {model_type} using datamodule properties: {e}", exc_info=True)
        # Depending on strictness, could raise an error or proceed with potentially incomplete HPs
        # For now, log and proceed.
    return final_hps


def _configure_trainer_core(
    training_config: dict[str, Any],
    callbacks_config: dict[str, bool],
    is_hpt_trial: bool = False,
    hpt_metric_to_monitor: str | None = "val_loss",
    optuna_trial_for_pruning: optuna.Trial | None = None,  # Keep parameter for compatibility
    checkpoint_dir_for_run: Path | None = None,
    log_dir_for_run: Path | None = None,
    model_type_for_paths: str | None = None,
    run_idx_for_paths: int | None = None,
) -> Result[pl.Trainer, str]:
    """
    Core helper to configure and create a PyTorch Lightning Trainer.
    """
    callbacks_list = []
    tb_logger = None  # type: ignore
    enable_model_checkpointing = False

    # 1. Early Stopping
    if callbacks_config.get("with_early_stopping", True):
        early_stopping_patience = training_config.get("early_stopping_patience", 10)
        monitor_metric = (
            hpt_metric_to_monitor if is_hpt_trial else training_config.get("early_stopping_monitor", "val_loss")
        )
        es_mode = "min" if "loss" in monitor_metric or "mse" in monitor_metric else "max"

        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric, patience=early_stopping_patience, verbose=True, mode=es_mode
        )
        callbacks_list.append(early_stopping_callback)

    # 2. Learning Rate Monitor - Add for both regular and HPT runs
    if callbacks_config.get("with_lr_monitor", True):
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        callbacks_list.append(lr_monitor_callback)

    # Check paths for both regular training and HPT trials
    if not checkpoint_dir_for_run or not log_dir_for_run or model_type_for_paths is None or run_idx_for_paths is None:
        return Failure("Missing path information for trainer configuration.")

    attempt_name = log_dir_for_run.name  # Assumes log_dir_for_run is the specific attempt path

    # 3. Model Checkpoint - Only for non-HPT runs
    if not is_hpt_trial and callbacks_config.get("with_model_checkpoint", True):
        enable_model_checkpointing = True
        checkpoint_filename = (
            f"{model_type_for_paths}-run{run_idx_for_paths}-{attempt_name}-{{epoch:02d}}-{{val_loss:.4f}}"
        )
        mc_monitor = training_config.get("checkpoint_monitor", "val_loss")
        mc_mode = "min" if "loss" in mc_monitor or "mse" in mc_monitor else "max"
        model_checkpoint_callback = ModelCheckpoint(
            monitor=mc_monitor,
            dirpath=str(checkpoint_dir_for_run),
            filename=checkpoint_filename,
            save_top_k=1,
            mode=mc_mode,
            save_last=True,
        )
        callbacks_list.append(model_checkpoint_callback)

    # 4. TensorBoard Logger - For both regular and HPT runs
    if callbacks_config.get("with_tensorboard_logger", True):
        try:
            tb_save_dir = log_dir_for_run.parent.parent.parent  # Up to .../logs/
            tb_name = f"{log_dir_for_run.parent.parent.name}/{log_dir_for_run.parent.name}"  # <model_type>/run_X
            tb_version = attempt_name

            tb_logger = TensorBoardLogger(
                save_dir=str(tb_save_dir),
                name=tb_name,
                version=tb_version,
                default_hp_metric=False,
            )
        except Exception as e:
            return Failure(f"Error setting up TensorBoardLogger paths: {e}")

    try:
        # Configure trainer arguments
        trainer_kwargs: dict[str, Any] = {
            "accelerator": training_config.get("accelerator", "auto"),
            "devices": training_config.get("devices", "auto"),
            "enable_progress_bar": training_config.get("enable_progress_bar", True),
            "callbacks": callbacks_list,
            "num_sanity_val_steps": training_config.get("num_sanity_val_steps", 2),
            "logger": tb_logger,  # Use logger for both regular and HPT
            "enable_checkpointing": enable_model_checkpointing,  # Only true for non-HPT
        }

        if is_hpt_trial:
            trainer_kwargs["max_epochs"] = training_config.get("max_epochs", 10)
            trainer_kwargs["enable_progress_bar"] = training_config.get("enable_progress_bar", False)
            trainer_kwargs["num_sanity_val_steps"] = training_config.get("num_sanity_val_steps", 0)
        else:  # Full run
            trainer_kwargs["max_epochs"] = training_config.get("max_epochs", 100)
            trainer_kwargs["reload_dataloaders_every_n_epochs"] = training_config.get(
                "reload_dataloaders_every_n_epochs", 1
            )
        # Create trainer
        trainer = pl.Trainer(**trainer_kwargs)
        return Success(trainer)
    except Exception as e:
        logger.error(f"Failed to create pl.Trainer: {e}", exc_info=True)
        return Failure(f"Trainer creation failed: {e}")


class ExperimentRunner:
    """
    Central runner class for hydrological model training experiments.
    Refactored to use centralized helper functions.
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        datamodule_config: dict[str, Any],
        training_config: dict[str, Any],
        num_runs: int = 1,
        base_seed: int = 42,
        override_previous_attempts: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.datamodule_config = datamodule_config
        self.training_config = training_config
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.override_previous_attempts = override_previous_attempts

        self.main_experiment_dir = self.output_dir / self.experiment_name
        try:
            self.main_experiment_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create main experiment directory: {e}")
            raise

    def _setup_model_directories(self, model_type: str) -> tuple[Path, Path]:
        model_checkpoints_base_dir = self.main_experiment_dir / checkpoint_manager.CHECKPOINTS_DIR_NAME / model_type
        model_logs_base_dir = self.main_experiment_dir / checkpoint_manager.LOGS_DIR_NAME / model_type
        try:
            model_checkpoints_base_dir.mkdir(parents=True, exist_ok=True)
            model_logs_base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directories for {model_type}: {e}")
            raise
        return model_checkpoints_base_dir, model_logs_base_dir

    def _get_hps_from_yaml(self, model_type: str, yaml_path: str) -> Result[dict[str, Any], str]:
        """Loads HPs from YAML, returns as Result."""
        try:
            model_hp = hp_from_yaml(model_type, yaml_path)
            return Success(model_hp)
        except Exception as e:
            logger.error(f"Failed to load hyperparameters from {yaml_path} for {model_type}: {e}")
            return Failure(f"HP loading failed for {model_type} from {yaml_path}: {e}")

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility across all libraries."""
        pl.seed_everything(seed, workers=True)

    def run_training_for_model(
        self,
        model_type: str,
        yaml_path: str,
        model_provider_fn: ModelProviderFn,
        gauge_ids: list[str],
    ) -> tuple[str | None, dict[str, Any]]:
        logger.info(f"Processing model: {model_type} using HPs from {yaml_path}")

        try:
            model_checkpoints_base_dir, model_logs_base_dir = self._setup_model_directories(model_type)

            # Load initial HPs from YAML
            initial_hps_result = self._get_hps_from_yaml(model_type, yaml_path)
            if isinstance(initial_hps_result, Failure):
                logger.error(initial_hps_result.failure())
                return None, {"error": initial_hps_result.failure()}
            initial_model_hps = initial_hps_result.unwrap()

            # Setup datamodule using core helper (HPs for datamodule come from initial_model_hps)
            datamodule_result = _setup_datamodule_core(
                base_datamodule_config=self.datamodule_config,
                hps_for_datamodule=initial_model_hps,  # YAML HPs guide DM config
                gauge_ids=gauge_ids,
                model_type=model_type,
            )
            if isinstance(datamodule_result, Failure):
                logger.error(datamodule_result.failure())
                return None, {"error": datamodule_result.failure()}
            datamodule = datamodule_result.unwrap()

            # Finalize model HPs using the now-configured datamodule
            finalized_model_hps = _finalize_model_hyperparameters(
                model_hps=initial_model_hps, datamodule=datamodule, model_type=model_type
            )
            logger.info(f"Finalized HPs for {model_type}: {finalized_model_hps}")

            current_model_run_results = []
            for run_idx in range(self.num_runs):
                logger.info(f"Starting run {run_idx + 1}/{self.num_runs} for {model_type}")
                current_run_seed = self.base_seed + run_idx
                self.set_seed(current_run_seed)
                logger.info(f"Using seed: {current_run_seed}")

                checkpoint_run_attempt_path_result = checkpoint_manager.determine_output_run_attempt_path(
                    base_model_output_dir=model_checkpoints_base_dir,
                    run_index=run_idx,
                    override_previous_attempts=self.override_previous_attempts,
                )
                if not isinstance(checkpoint_run_attempt_path_result, Success):
                    logger.error(f"Failed to determine checkpoint path: {checkpoint_run_attempt_path_result.failure()}")
                    continue
                specific_checkpoint_dir = checkpoint_run_attempt_path_result.unwrap()

                log_run_attempt_path_result = checkpoint_manager.determine_log_run_attempt_path(
                    base_model_log_dir=model_logs_base_dir,
                    run_index=run_idx,
                    override_previous_attempts=self.override_previous_attempts,
                )
                if not isinstance(log_run_attempt_path_result, Success):
                    logger.error(f"Failed to determine log path: {log_run_attempt_path_result.failure()}")
                    continue
                specific_log_dir = log_run_attempt_path_result.unwrap()

                # Configure trainer using core helper
                callbacks_config_full_run = {
                    "with_early_stopping": True,
                    "with_model_checkpoint": True,
                    "with_lr_monitor": True,
                    "with_tensorboard_logger": True,
                    "with_optuna_pruning": False,  # Not an HPT trial
                }
                trainer_result = _configure_trainer_core(
                    training_config=self.training_config,
                    callbacks_config=callbacks_config_full_run,
                    is_hpt_trial=False,
                    checkpoint_dir_for_run=specific_checkpoint_dir,
                    log_dir_for_run=specific_log_dir,
                    model_type_for_paths=model_type,
                    run_idx_for_paths=run_idx,
                )
                if isinstance(trainer_result, Failure):
                    logger.error(f"Trainer config error for {model_type}, run {run_idx}: {trainer_result.failure()}")
                    continue
                trainer = trainer_result.unwrap()

                try:
                    # Model provider function now receives the finalized HPs
                    model = model_provider_fn(model_type, copy.deepcopy(finalized_model_hps))
                except Exception as e:
                    logger.error(f"Failed to provide model for {model_type}, run {run_idx}: {e}", exc_info=True)
                    continue

                status = "failed"
                try:
                    trainer.fit(model, datamodule=datamodule)
                    # Assuming ModelCheckpoint is always the second callback if present and not HPT
                    mc_callback = None
                    for cb in trainer.callbacks:  # type: ignore
                        if isinstance(cb, ModelCheckpoint):
                            mc_callback = cb
                            break

                    if mc_callback and mc_callback.best_model_path and mc_callback.best_model_score is not None:
                        best_model_path = mc_callback.best_model_path
                        best_model_score_value = mc_callback.best_model_score.item()
                        logger.info(
                            f"Run {run_idx} completed. Best val_loss: {best_model_score_value}, Path: {best_model_path}"
                        )
                        current_model_run_results.append(
                            (
                                best_model_path,
                                {"val_loss": best_model_score_value, "run_index": run_idx, "seed": current_run_seed},
                            )
                        )
                        status = "success"
                    else:
                        logger.warning(f"No best model path or score found from ModelCheckpoint for run {run_idx}")
                except Exception as e:
                    logger.error(f"Training failed for {model_type}, run {run_idx}: {e}", exc_info=True)
                finally:
                    if hasattr(trainer, "logger") and trainer.logger:
                        trainer.logger.finalize(status)  # type: ignore

            return self._process_results(model_type, current_model_run_results, model_checkpoints_base_dir)

        except Exception as e:
            logger.error(f"Unexpected error in run_training_for_model for {model_type}: {e}", exc_info=True)
            return None, {"error": str(e)}

    def _process_results(
        self,
        model_type: str,
        model_run_results: list[tuple[str, dict[str, Any]]],
        model_checkpoints_base_dir: Path,
    ) -> tuple[str | None, dict[str, Any]]:
        if not model_run_results:
            logger.warning(f"No successful runs for {model_type}")
            return None, {"error": "No successful runs"}

        sorted_results = sorted(model_run_results, key=lambda x: x[1]["val_loss"])  # Assumes val_loss is the metric
        overall_best_model_path, overall_best_metrics = sorted_results[0]

        try:
            relative_best_path = Path(overall_best_model_path).relative_to(model_checkpoints_base_dir)
            update_result = checkpoint_manager.update_overall_best_model_info_file(
                model_checkpoints_output_dir=model_checkpoints_base_dir,
                best_checkpoint_relative_path=str(relative_best_path),
            )
            if not isinstance(update_result, Success):
                logger.error(f"Failed to update best model info for {model_type}: {update_result.failure()}")
            logger.info(f"Overall best model for {model_type}: {relative_best_path}, Metrics: {overall_best_metrics}")
            return overall_best_model_path, overall_best_metrics
        except Exception as e:
            logger.error(f"Error recording overall best model for {model_type}: {e}")
            return None, {"error": f"Error recording best model: {str(e)}"}

    def run_experiment(
        self,
        model_types: list[str],
        yaml_paths: list[str],
        model_provider_fns: list[ModelProviderFn],
        gauge_ids: list[str],
    ) -> Result[dict[str, tuple[str | None, dict[str, Any]]], str]:
        if not gauge_ids:
            return Failure("No gauge IDs provided for training")
        if len(model_types) != len(yaml_paths) or len(model_types) != len(model_provider_fns):
            return Failure("Length mismatch: model_types, yaml_paths, model_provider_fns")

        logger.info(f"Starting experiment '{self.experiment_name}' from ExperimentRunner.")
        all_models_results = {}
        for i, (current_model_type, current_yaml_path, current_provider_fn) in enumerate(
            zip(model_types, yaml_paths, model_provider_fns, strict=True)
        ):
            logger.info(f"Processing model ({i + 1}/{len(model_types)}): {current_model_type}")
            try:
                best_result = self.run_training_for_model(
                    model_type=current_model_type,
                    yaml_path=current_yaml_path,
                    model_provider_fn=current_provider_fn,
                    gauge_ids=gauge_ids,
                )
                all_models_results[current_model_type] = best_result
            except Exception as e:
                logger.error(f"Failed to process model {current_model_type} in run_experiment: {e}", exc_info=True)
                all_models_results[current_model_type] = (None, {"error": str(e)})

        if not all_models_results:
            return Failure("No models were successfully processed in experiment.")
        return Success(all_models_results)
