import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

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
ModelProviderFn = Callable[[str, str], ModelType]


class ExperimentRunner:
    """
    Central runner class for hydrological model training experiments.

    This class abstracts common functionality between training from scratch and fine-tuning,
    while allowing flexibility in how models are provided (created or loaded).
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
        Initialize the experiment runner.

        Args:
            output_dir: Base directory for storing outputs (checkpoints, logs)
            experiment_name: Name of the experiment for directory organization
            datamodule_config: Configuration for HydroInMemoryDataModule
            training_config: Configuration for PyTorch Lightning Trainer
            num_runs: Number of independent training runs for each model type
            base_seed: Base random seed (will be incremented for each run)
            override_previous_attempts: Whether to override existing outputs
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.datamodule_config = datamodule_config
        self.training_config = training_config
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.override_previous_attempts = override_previous_attempts

        # Create main experiment directory
        self.main_experiment_dir = self.output_dir / self.experiment_name
        try:
            self.main_experiment_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create main experiment directory: {e}")
            raise

    def _setup_model_directories(self, model_type: str) -> tuple[Path, Path]:
        """
        Set up directories for a specific model type.

        Args:
            model_type: Type of model ('tide', 'tft', 'ealstm', etc.)

        Returns:
            Tuple of paths (checkpoints_dir, logs_dir)
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

    def _setup_datamodule(
        self, model_type: str, yaml_path: str, gauge_ids: list[str]
    ) -> tuple[HydroInMemoryDataModule, dict[str, Any]]:
        """
        Set up data module based on model hyperparameters.

        Args:
            model_type: Type of model ('tide', 'tft', 'ealstm', etc.)
            yaml_path: Path to the YAML file with model hyperparameters
            gauge_ids: List of basin/gauge IDs to use for training

        Returns:
            Tuple of (datamodule, model_hp)
        """
        # Load model hyperparameters
        model_hp = hp_from_yaml(model_type, yaml_path)
        input_length = model_hp.get("input_len")
        output_length = model_hp.get("output_len")

        if input_length is None or output_length is None:
            raise ValueError(f"Missing input_len or output_len in hyperparameters for {model_type}")

        # Create a copy of datamodule_config with the correct input/output lengths
        current_datamodule_config = self.datamodule_config.copy()
        current_datamodule_config["input_length"] = input_length
        current_datamodule_config["output_length"] = output_length

        # Set up the data module
        datamodule = HydroInMemoryDataModule(list_of_gauge_ids_to_process=gauge_ids, **current_datamodule_config)

        # Verify setup
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule, model_hp

    def _configure_trainer_and_callbacks(
        self,
        model_type: str,
        run_idx: int,
        specific_checkpoint_dir: Path,
        specific_log_dir: Path,
    ) -> tuple[pl.Trainer, ModelCheckpoint, str]:
        """
        Configure PyTorch Lightning Trainer and callbacks.

        Args:
            model_type: Type of model ('tide', 'tft', 'ealstm', etc.)
            run_idx: Index of the current run
            specific_checkpoint_dir: Directory to save checkpoints for this run/attempt
            specific_log_dir: Directory to save logs for this run/attempt

        Returns:
            Tuple of (trainer, checkpoint_callback, attempt_name)
        """
        attempt_name = specific_log_dir.name

        # Set up checkpoint callback
        checkpoint_filename = f"{model_type}-run{run_idx}-{attempt_name}-{{epoch:02d}}-{{val_loss:.4f}}"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(specific_checkpoint_dir),
            filename=checkpoint_filename,
            save_top_k=1,
            mode="min",
            save_last=True,
        )

        # Set up early stopping callback
        early_stopping_patience = self.training_config.get("early_stopping_patience", 10)
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            verbose=True,
            mode="min",
        )

        # Set up learning rate monitor
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")

        # Set up TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(specific_log_dir.parent.parent),  # Navigate up to the logs base dir
            name=f"run_{run_idx}",
            version=attempt_name,
            default_hp_metric=False,
        )

        callbacks_list = [early_stopping_callback, checkpoint_callback, lr_monitor_callback]

        # Configure trainer arguments
        trainer_kwargs = {
            "max_epochs": self.training_config.get("max_epochs", 100),
            "accelerator": self.training_config.get("accelerator", "auto"),
            "devices": self.training_config.get("devices", 1),
            "enable_progress_bar": self.training_config.get("enable_progress_bar", True),
            "logger": tb_logger,
            "callbacks": callbacks_list,
            "num_sanity_val_steps": self.training_config.get("num_sanity_val_steps", 2),
            "reload_dataloaders_every_n_epochs": self.training_config.get("reload_dataloaders_every_n_epochs", 1),
        }

        # Configure precision if specified
        precision = self.training_config.get("precision")
        if precision is not None:
            valid_precision_values = ["32-true", "32", "16-mixed", "bf16-mixed"]
            if str(precision) in valid_precision_values:
                trainer_kwargs["precision"] = precision
            else:
                logger.warning(f"Invalid precision value: {precision}. Using default.")

        # Configure matmul precision if specified
        matmul_precision = self.training_config.get("torch_float32_matmul_precision")
        if matmul_precision is not None:
            valid_matmul_values = ["highest", "high", "medium"]
            if matmul_precision in valid_matmul_values:
                torch.set_float32_matmul_precision(matmul_precision)
            else:
                logger.warning(f"Invalid torch_float32_matmul_precision value: {matmul_precision}. Not setting.")

        # Create trainer
        trainer = pl.Trainer(**trainer_kwargs)

        return trainer, checkpoint_callback, attempt_name

    def _process_results(
        self,
        model_type: str,
        model_run_results: list[tuple[str, dict[str, Any]]],
        model_checkpoints_base_dir: Path,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Process results from multiple runs of a single model type.

        Args:
            model_type: Type of model ('tide', 'tft', 'ealstm', etc.)
            model_run_results: List of tuples (checkpoint_path, metrics_dict) for all runs
            model_checkpoints_base_dir: Base directory for checkpoints for this model type

        Returns:
            Tuple of (best_checkpoint_path, best_metrics_dict)
        """
        if not model_run_results:
            logger.warning(f"No successful runs for {model_type}")
            return None, {"error": "No successful runs"}

        # Sort by validation loss (ascending)
        sorted_results = sorted(model_run_results, key=lambda x: x[1]["val_loss"])
        overall_best_model_path, overall_best_metrics = sorted_results[0]

        try:
            # Get relative path for overall best model
            relative_best_path = Path(overall_best_model_path).relative_to(model_checkpoints_base_dir)

            # Update overall best model info file
            update_result = checkpoint_manager.update_overall_best_model_info_file(
                model_checkpoints_output_dir=model_checkpoints_base_dir,
                best_checkpoint_relative_path=str(relative_best_path),
            )

            if not isinstance(update_result, Success):
                logger.error(f"Failed to update best model info for {model_type}: {update_result.failure()}")

            logger.info(f"Overall best model for {model_type}: {relative_best_path}")
            logger.info(f"Best metrics: {overall_best_metrics}")

            return overall_best_model_path, overall_best_metrics

        except Exception as e:
            logger.error(f"Error recording overall best model for {model_type}: {e}")
            return None, {"error": f"Error recording best model: {str(e)}"}

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
        """
        Run training for a specific model type using the provided model provisioning function.

        Args:
            model_type: Type of model ('tide', 'tft', 'ealstm', etc.)
            yaml_path: Path to the YAML file with model hyperparameters
            model_provider_fn: Function that takes (model_type, yaml_path) and returns a model
            gauge_ids: List of basin/gauge IDs to use for training

        Returns:
            Tuple of (best_checkpoint_path, best_metrics_dict)
        """
        logger.info(f"Processing model: {model_type}")

        try:
            # 1. Setup directories
            model_checkpoints_base_dir, model_logs_base_dir = self._setup_model_directories(model_type)

            # 2. Setup datamodule
            datamodule, _ = self._setup_datamodule(model_type, yaml_path, gauge_ids)

            # 3. Collection for run results
            current_model_run_results = []

            # 4. Execute multiple runs
            for run_idx in range(self.num_runs):
                logger.info(f"Starting run {run_idx + 1}/{self.num_runs} for {model_type}")

                # 4a. Set seed for reproducibility
                current_run_seed = self.base_seed + run_idx
                self.set_seed(current_run_seed)
                logger.info(f"Using seed: {current_run_seed}")

                # 4b. Determine paths for this run/attempt
                checkpoint_run_attempt_path_result = checkpoint_manager.determine_output_run_attempt_path(
                    base_model_output_dir=model_checkpoints_base_dir,
                    run_index=run_idx,
                    override_previous_attempts=self.override_previous_attempts,
                )

                if not isinstance(checkpoint_run_attempt_path_result, Success):
                    logger.error(
                        f"Failed to determine checkpoint path for run {run_idx}: {checkpoint_run_attempt_path_result.failure()}"
                    )
                    continue

                specific_checkpoint_dir = checkpoint_run_attempt_path_result.unwrap()

                log_run_attempt_path_result = checkpoint_manager.determine_log_run_attempt_path(
                    base_model_log_dir=model_logs_base_dir,
                    run_index=run_idx,
                    override_previous_attempts=self.override_previous_attempts,
                )

                if not isinstance(log_run_attempt_path_result, Success):
                    logger.error(
                        f"Failed to determine log path for run {run_idx}: {log_run_attempt_path_result.failure()}"
                    )
                    continue

                specific_log_dir = log_run_attempt_path_result.unwrap()

                # 4c. Configure trainer and callbacks
                try:
                    trainer, checkpoint_callback, attempt_name = self._configure_trainer_and_callbacks(
                        model_type=model_type,
                        run_idx=run_idx,
                        specific_checkpoint_dir=specific_checkpoint_dir,
                        specific_log_dir=specific_log_dir,
                    )
                except Exception as e:
                    logger.error(f"Failed to configure trainer for {model_type}, run {run_idx}: {e}")
                    continue

                # 4d. Get model from provider function
                try:
                    model = model_provider_fn(model_type, yaml_path)
                except Exception as e:
                    logger.error(f"Failed to provide model for {model_type}, run {run_idx}: {e}")
                    continue

                # 4e. Execute training
                status = "failed"
                try:
                    trainer.fit(model, datamodule=datamodule)

                    # Collect results
                    best_model_path = checkpoint_callback.best_model_path
                    best_model_score = checkpoint_callback.best_model_score

                    if best_model_path and best_model_score is not None:
                        best_model_score_value = best_model_score.item()
                        logger.info(f"Run {run_idx} completed with best val_loss: {best_model_score_value}")

                        # Store the result for this run
                        current_model_run_results.append(
                            (
                                best_model_path,
                                {"val_loss": best_model_score_value, "run_index": run_idx, "seed": current_run_seed},
                            )
                        )
                        status = "success"
                    else:
                        logger.warning(f"No best model path or score found for run {run_idx}")

                except Exception as e:
                    logger.error(f"Training failed for {model_type}, run {run_idx}: {e}")

                finally:
                    if hasattr(trainer, "logger") and trainer.logger:
                        trainer.logger.finalize(status)

            # 5. Process results across all runs
            return self._process_results(
                model_type=model_type,
                model_run_results=current_model_run_results,
                model_checkpoints_base_dir=model_checkpoints_base_dir,
            )

        except Exception as e:
            logger.error(f"Unexpected error in run_training_for_model for {model_type}: {e}")
            return None, {"error": str(e)}

    def run_experiment(
        self,
        model_types: list[str],
        yaml_paths: list[str],
        model_provider_fns: list[ModelProviderFn],
        gauge_ids: list[str],
    ) -> Result[dict[str, tuple[str | None, dict[str, Any]]], str]:
        """
        Run training experiment for multiple model types.

        Args:
            model_types: List of model types to train ('tide', 'tft', 'ealstm', etc.)
            yaml_paths: List of paths to YAML files with model hyperparameters
            model_provider_fns: List of functions that provide models (one per model type)
            gauge_ids: List of basin/gauge IDs to use for training

        Returns:
            Result containing a dictionary mapping model types to tuples of
            (best_checkpoint_path, metrics_dict) if successful, or an error message if failed.
        """
        if not gauge_ids:
            return Failure("No gauge IDs provided for training")

        if len(model_types) != len(yaml_paths) or len(model_types) != len(model_provider_fns):
            return Failure("Length mismatch between model_types, yaml_paths, and model_provider_fns")

        logger.info(f"Starting experiment '{self.experiment_name}'")
        logger.info(f"Output directory: {self.main_experiment_dir}")
        logger.info(f"Models to process: {', '.join(model_types)}")
        logger.info(f"Number of runs per model: {self.num_runs}")

        all_models_results = {}

        # Process each model type
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
                logger.error(f"Failed to process model {current_model_type}: {e}")
                all_models_results[current_model_type] = (None, {"error": str(e)})

        if not all_models_results:
            return Failure("No models were successfully processed")

        return Success(all_models_results)
