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

from ..data.in_memory_datamodule import HydroInMemoryDataModule
from ..exceptions import ConfigurationError, DataProcessingError, FileOperationError, ModelTrainingError
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
) -> HydroInMemoryDataModule:
    """
    Core helper to configure and set up HydroInMemoryDataModule.

    Args:
        base_datamodule_config: Base configuration for the datamodule
        hps_for_datamodule: Hyperparameters for the datamodule
        gauge_ids: List of gauge IDs to process
        model_type: Type of model being configured

    Returns:
        Configured HydroInMemoryDataModule instance

    Raises:
        DataProcessingError: If datamodule setup fails
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
        return datamodule
    except Exception as e:
        logger.error(f"Failed to setup datamodule for {model_type}: {e}", exc_info=True)
        raise DataProcessingError(f"Datamodule setup failed for {model_type}: {e}") from e


def _finalize_model_hyperparameters(
    model_hps: dict[str, Any],
    datamodule: HydroInMemoryDataModule,
    model_type: str,  # Added for logging context
) -> dict[str, Any]:
    """
    Enriches model HPs with datamodule-derived dimensions and other common mappings.

    Args:
        model_hps: Base model hyperparameters
        datamodule: Configured datamodule instance
        model_type: Type of model being configured

    Returns:
        Finalized hyperparameters dictionary with datamodule-derived properties
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
) -> pl.Trainer:
    """
    Core helper to configure and create a PyTorch Lightning Trainer.

    Args:
        training_config: Training configuration dictionary
        callbacks_config: Callbacks configuration dictionary
        is_hpt_trial: Whether this is a hyperparameter tuning trial
        hpt_metric_to_monitor: Metric to monitor for HPT trials
        optuna_trial_for_pruning: Optuna trial for pruning (kept for compatibility)
        checkpoint_dir_for_run: Directory for checkpoints
        log_dir_for_run: Directory for logs
        model_type_for_paths: Model type for path construction
        run_idx_for_paths: Run index for path construction

    Returns:
        Configured PyTorch Lightning Trainer

    Raises:
        ConfigurationError: If required path information is missing
        ModelTrainingError: If trainer creation fails
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
        raise ConfigurationError("Missing path information for trainer configuration.")

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
            raise ConfigurationError(f"Error setting up TensorBoardLogger paths: {e}") from e

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
        return trainer
    except Exception as e:
        logger.error(f"Failed to create pl.Trainer: {e}", exc_info=True)
        raise ModelTrainingError(f"Trainer creation failed: {e}") from e


class ExperimentRunner:
    """
    Central runner class for hydrological model training experiments.
    Refactored to use centralized helper functions and exception-based error handling.

    Raises:
        ConfigurationError: For configuration and input validation errors
        FileOperationError: For file operation errors
        DataProcessingError: For data processing and datamodule setup errors
        ModelTrainingError: For model training and trainer configuration errors
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
        """
        Initialize the ExperimentRunner.

        Args:
            output_dir: Directory for experiment outputs
            experiment_name: Name of the experiment
            datamodule_config: Configuration for the datamodule
            training_config: Configuration for training
            num_runs: Number of training runs per model
            base_seed: Base seed for reproducibility
            override_previous_attempts: Whether to override previous attempts

        Raises:
            Exception: If main experiment directory creation fails
        """
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
        """
        Set up directories for model checkpoints and logs.

        Args:
            model_type: Type of model to create directories for

        Returns:
            Tuple of (checkpoints_dir, logs_dir)

        Raises:
            Exception: If directory creation fails
        """
        model_checkpoints_base_dir = self.main_experiment_dir / checkpoint_manager.CHECKPOINTS_DIR_NAME / model_type
        model_logs_base_dir = self.main_experiment_dir / checkpoint_manager.LOGS_DIR_NAME / model_type
        try:
            model_checkpoints_base_dir.mkdir(parents=True, exist_ok=True)
            model_logs_base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directories for {model_type}: {e}")
            raise
        return model_checkpoints_base_dir, model_logs_base_dir

    def _get_hps_from_yaml(self, model_type: str, yaml_path: str) -> dict[str, Any]:
        """
        Loads hyperparameters from YAML file.

        Args:
            model_type: Type of model to load HPs for
            yaml_path: Path to the YAML file

        Returns:
            Dictionary of hyperparameters

        Raises:
            FileOperationError: If HP loading fails
        """
        try:
            model_hp = hp_from_yaml(model_type, yaml_path)
            return model_hp
        except Exception as e:
            logger.error(f"Failed to load hyperparameters from {yaml_path} for {model_type}: {e}")
            raise FileOperationError(f"HP loading failed for {model_type} from {yaml_path}: {e}") from e

    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility across all libraries.

        Args:
            seed: Random seed value
        """
        from .seed_manager import SeedManager

        seed_manager = SeedManager(seed)
        seed_manager.set_global_seeds()

    def run_training_for_model(
        self,
        model_type: str,
        yaml_path: str,
        model_provider_fn: ModelProviderFn,
        gauge_ids: list[str],
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Run training for a specific model type.

        Args:
            model_type: Type of model to train
            yaml_path: Path to YAML configuration file
            model_provider_fn: Function to provide model instance
            gauge_ids: List of gauge IDs for training

        Returns:
            Tuple of (best_model_path, metrics_dict) or (None, error_dict)

        Raises:
            FileOperationError: If HP loading fails
            DataProcessingError: If datamodule setup fails
            ConfigurationError: If trainer configuration fails
            ModelTrainingError: If training fails
        """
        logger.info(f"Processing model: {model_type} using HPs from {yaml_path}")

        try:
            model_checkpoints_base_dir, model_logs_base_dir = self._setup_model_directories(model_type)

            # Load initial HPs from YAML
            try:
                initial_model_hps = self._get_hps_from_yaml(model_type, yaml_path)
            except FileOperationError as e:
                logger.error(str(e))
                return None, {"error": str(e)}

            # Setup datamodule using core helper (HPs for datamodule come from initial_model_hps)
            try:
                datamodule = _setup_datamodule_core(
                    base_datamodule_config=self.datamodule_config,
                    hps_for_datamodule=initial_model_hps,  # YAML HPs guide DM config
                    gauge_ids=gauge_ids,
                    model_type=model_type,
                )
            except DataProcessingError as e:
                logger.error(str(e))
                return None, {"error": str(e)}

            # Finalize model HPs using the now-configured datamodule
            finalized_model_hps = _finalize_model_hyperparameters(
                model_hps=initial_model_hps, datamodule=datamodule, model_type=model_type
            )
            logger.info(f"Finalized HPs for {model_type}: {finalized_model_hps}")

            current_model_run_results = []
            for run_idx in range(self.num_runs):
                logger.info(f"Starting run {run_idx + 1}/{self.num_runs} for {model_type}")
                from .seed_manager import SeedManager

                base_seed_manager = SeedManager(self.base_seed)
                current_run_seed = base_seed_manager.get_operation_seed("model_training", f"run_{run_idx}")
                self.set_seed(current_run_seed)
                logger.info(
                    f"Using seed: {current_run_seed} (derived from base seed: {self.base_seed}, run: {run_idx})"
                )

                try:
                    specific_checkpoint_dir = checkpoint_manager.determine_output_run_attempt_path(
                        base_model_output_dir=model_checkpoints_base_dir,
                        run_index=run_idx,
                        override_previous_attempts=self.override_previous_attempts,
                    )

                    specific_log_dir = checkpoint_manager.determine_log_run_attempt_path(
                        base_model_log_dir=model_logs_base_dir,
                        run_index=run_idx,
                        override_previous_attempts=self.override_previous_attempts,
                    )
                except Exception as e:
                    logger.error(f"Failed to determine paths for run {run_idx}: {e}")
                    continue

                # Configure trainer using core helper
                try:
                    callbacks_config_full_run = {
                        "with_early_stopping": True,
                        "with_model_checkpoint": True,
                        "with_lr_monitor": True,
                        "with_tensorboard_logger": True,
                        "with_optuna_pruning": False,  # Not an HPT trial
                    }
                    trainer = _configure_trainer_core(
                        training_config=self.training_config,
                        callbacks_config=callbacks_config_full_run,
                        is_hpt_trial=False,
                        checkpoint_dir_for_run=specific_checkpoint_dir,
                        log_dir_for_run=specific_log_dir,
                        model_type_for_paths=model_type,
                        run_idx_for_paths=run_idx,
                    )
                except (ConfigurationError, ModelTrainingError) as e:
                    logger.error(f"Trainer config error for {model_type}, run {run_idx}: {e}")
                    continue

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
        """
        Process training results and determine the best model.

        Args:
            model_type: Type of model being processed
            model_run_results: List of (model_path, metrics) tuples
            model_checkpoints_base_dir: Base directory for model checkpoints

        Returns:
            Tuple of (best_model_path, metrics_dict) or (None, error_dict)
        """
        if not model_run_results:
            logger.warning(f"No successful runs for {model_type}")
            return None, {"error": "No successful runs"}

        sorted_results = sorted(model_run_results, key=lambda x: x[1]["val_loss"])  # Assumes val_loss is the metric
        overall_best_model_path, overall_best_metrics = sorted_results[0]

        try:
            relative_best_path = Path(overall_best_model_path).relative_to(model_checkpoints_base_dir)
            checkpoint_manager.update_overall_best_model_info_file(
                model_checkpoints_output_dir=model_checkpoints_base_dir,
                best_checkpoint_relative_path=str(relative_best_path),
            )
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
    ) -> dict[str, tuple[str | None, dict[str, Any]]]:
        """
        Run training experiment for multiple model types.

        Args:
            model_types: List of model types to train
            yaml_paths: List of YAML configuration file paths
            model_provider_fns: List of model provider functions
            gauge_ids: List of gauge IDs for training

        Returns:
            Dictionary mapping model types to (best_model_path, metrics_dict) tuples

        Raises:
            ConfigurationError: If input validation fails
        """
        if not gauge_ids:
            raise ConfigurationError("No gauge IDs provided for training")
        if len(model_types) != len(yaml_paths) or len(model_types) != len(model_provider_fns):
            raise ConfigurationError("Length mismatch: model_types, yaml_paths, model_provider_fns")

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
            raise ConfigurationError("No models were successfully processed in experiment.")
        return all_models_results
