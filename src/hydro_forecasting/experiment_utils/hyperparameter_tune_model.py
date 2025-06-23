import gc
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml

from hydro_forecasting.data.in_memory_datamodule import HydroInMemoryDataModule
from hydro_forecasting.experiment_utils import checkpoint_manager
from hydro_forecasting.models import model_factory

from ..exceptions import ConfigurationError, FileOperationError, ModelTrainingError
from .training_runner import _configure_trainer_core, _finalize_model_hyperparameters, _setup_datamodule_core

logger = logging.getLogger(__name__)


def _handle_trial_error(
    error_message: str,
    trial: optuna.Trial,
) -> float:
    logger.error(f"Trial {trial.number}: {error_message}")
    raise optuna.exceptions.TrialPruned(error_message)


def _load_search_space(model_type: str, search_spaces_dir: str | Path = "search_spaces") -> dict[str, Any]:
    """
    Get the expected parameters for a specified model type.

    Args:
        model_type: The type of model (tide, tft, ealstm, tsmixer)
        search_spaces_dir: Directory containing search space definitions

    Returns:
        The search space dictionary

    Raises:
        ConfigurationError: If search space file is not found or cannot be loaded
        FileOperationError: If file operations fail
    """
    search_spaces_path = Path(search_spaces_dir)
    model_space_file = search_spaces_path / f"{model_type.lower()}_space.py"
    if not model_space_file.exists():
        raise FileOperationError(f"Search space file not found: {model_space_file}")
    try:
        spec = importlib.util.spec_from_file_location(f"search_spaces.{model_type.lower()}_space", model_space_file)
        if spec is None or spec.loader is None:
            raise ConfigurationError(f"Could not create module spec for {model_space_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if not hasattr(module, "get_search_space"):
            raise ConfigurationError(f"'get_search_space' function not found in {model_space_file}")
        return module.get_search_space()
    except ImportError as e:
        raise ConfigurationError(f"Error importing search space for {model_type}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Unexpected error loading search space for {model_type}: {e}")


def _suggest_trial_hparams(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """
    Suggest hyperparameters for a trial based on the search space.

    Args:
        trial: Optuna trial
        search_space: Search space dictionary

    Returns:
        Dictionary of suggested hyperparameters
    """
    hparams: dict[str, Any] = {}
    combined_search_space: dict[str, Any] = {}
    if "common" in search_space:
        combined_search_space.update(search_space["common"])
    if "model_specific" in search_space:
        combined_search_space.update(search_space["model_specific"])

    for param_name, config in combined_search_space.items():
        param_type = config["type"]
        if param_type == "categorical":
            hparams[param_name] = trial.suggest_categorical(param_name, config["choices"])
        elif param_type == "int":
            hparams[param_name] = trial.suggest_int(
                param_name, config["low"], config["high"], step=config.get("step", 1), log=config.get("log", False)
            )
        elif param_type == "float":
            hparams[param_name] = trial.suggest_float(
                param_name, config["low"], config["high"], step=config.get("step"), log=config.get("log", False)
            )
    return hparams


def save_best_hp_to_yaml(
    best_config: dict[str, Any],
    output_dir: Path,
    model_type: str,
) -> Path:
    """
    Save the best hyperparameters to a YAML file.

    Args:
        best_config: Best hyperparameters configuration
        output_dir: Directory to save the YAML file
        model_type: Type of model being tuned

    Returns:
        Path to the saved YAML file

    Raises:
        FileOperationError: If file operations fail
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output file path
        yaml_path = output_dir / f"{model_type}_best_params.yaml"

        # Write to YAML file
        with open(yaml_path, "w") as f:
            yaml.safe_dump(best_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved best hyperparameters to {yaml_path}")
        return yaml_path

    except Exception as e:
        logger.error(f"Error saving best HP to YAML: {e}", exc_info=True)
        raise FileOperationError(f"Failed to save best hyperparameters to YAML: {e}")


def plot_optimization_history(
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
) -> Path:
    """
    Create and save optimization history plot.

    Args:
        study: Completed optuna study
        output_dir: Directory to save the plot
        model_type: Type of model being tuned

    Returns:
        Path to the saved plot

    Raises:
        FileOperationError: If file operations fail
    """
    try:
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get dataframe of all trials
        df = study.trials_dataframe()

        # Plot optimization history
        plt.figure(figsize=(10, 6))
        plt.title(f"Optimization History - {model_type}")
        plt.xlabel("Trial Number")
        plt.ylabel(f"Objective Value ({study.direction})")
        plt.plot(df["number"], df["value"], "o-")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        history_path = plots_dir / f"{model_type}_optimization_history.png"
        plt.savefig(history_path)
        plt.close()

        logger.info(f"Saved optimization history plot to {history_path}")
        return history_path

    except Exception as e:
        logger.error(f"Error creating optimization history plot: {e}", exc_info=True)
        raise FileOperationError(f"Failed to create optimization history plot: {e}")


def plot_parameter_importance(
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
) -> Path:
    """
    Create and save parameter importance plot.

    Args:
        study: Completed optuna study
        output_dir: Directory to save the plot
        model_type: Type of model being tuned

    Returns:
        Path to the saved plot

    Raises:
        ConfigurationError: If there are not enough trials for importance analysis
        FileOperationError: If file operations fail
    """
    try:
        # Check if there are enough trials for importance analysis
        if len(study.trials) < 5:
            raise ConfigurationError("Not enough trials (minimum 5) to calculate parameter importance")

        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Calculate parameter importances
        importances = optuna.importance.get_param_importances(study)
        importance_df = pd.DataFrame(importances.items(), columns=["Parameter", "Importance"]).sort_values(
            "Importance", ascending=True
        )

        # Plot parameter importances
        plt.figure(figsize=(10, max(6, len(importances) * 0.3)))
        plt.barh(importance_df["Parameter"], importance_df["Importance"])
        plt.title(f"Parameter Importances - {model_type}")
        plt.xlabel("Importance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        importance_path = plots_dir / f"{model_type}_parameter_importance.png"
        plt.savefig(importance_path)
        plt.close()

        logger.info(f"Saved parameter importance plot to {importance_path}")
        return importance_path

    except Exception as e:
        logger.error(f"Error creating parameter importance plot: {e}", exc_info=True)
        raise FileOperationError(f"Failed to create parameter importance plot: {e}")


def analyze_hpt_results(
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
    save_yaml: bool = True,
    create_plots: bool = True,
) -> dict[str, Any]:
    """
    Analyze hyperparameter tuning results, save best parameters to YAML,
    and create visualization plots.

    Args:
        study: The completed optuna study
        output_dir: Directory to save results
        model_type: Type of model being tuned
        save_yaml: Whether to save best params to YAML
        create_plots: Whether to create visualization plots

    Returns:
        The best hyperparameters configuration

    Raises:
        ConfigurationError: If no trials found in study
        FileOperationError: If file operations fail during analysis
    """
    try:
        if not study.trials:
            raise ConfigurationError("No trials found in study.")

        # Get best trial and its parameters
        best_trial = study.best_trial
        best_params = best_trial.params

        # Get complete model config from user_attrs if available
        best_config = best_params.copy()
        if "model_config" in best_trial.user_attrs:
            best_config = best_trial.user_attrs["model_config"]
        elif "hparams" in best_trial.user_attrs:
            best_config = best_trial.user_attrs["hparams"]

        # Log results summary
        logger.info(f"HPT completed for {model_type}. Best trial: #{best_trial.number}")
        logger.info(f"Best {study.direction} value: {best_trial.value}")
        logger.info("Best hyperparameters:")
        for param_name, param_value in best_config.items():
            logger.info(f"  {param_name}: {param_value}")

        # Save best hyperparameters to YAML if requested
        if save_yaml:
            try:
                save_best_hp_to_yaml(best_config, output_dir, model_type)
            except FileOperationError as e:
                logger.warning(f"Failed to save best HP to YAML: {e}")

        # Create visualization plots if requested
        if create_plots and len(study.trials) > 1:
            try:
                plot_optimization_history(study, output_dir, model_type)
            except FileOperationError as e:
                logger.warning(f"Failed to create optimization history plot: {e}")

            if len(study.trials) >= 5:
                try:
                    plot_parameter_importance(study, output_dir, model_type)
                except (ConfigurationError, FileOperationError) as e:
                    logger.warning(f"Failed to create parameter importance plot: {e}")
            else:
                logger.info("Not enough trials (minimum 5) to create parameter importance plot")

        return best_config

    except Exception as e:
        logger.error(f"Error analyzing HPT results: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to analyze HPT results: {e}")


def hyperparameter_tune_model(
    model_type: str,
    gauge_ids: list[str],
    datamodule_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir_study: str | Path,
    experiment_name: str,
    n_trials: int,
    search_spaces_dir: str | Path = "search_spaces",
    optuna_storage_url: str | None = None,
    optuna_direction: str = "minimize",
    optuna_sampler: optuna.samplers.BaseSampler | None = None,
    # Removed optuna_pruner parameter
    metric_to_optimize: str = "val_loss",
    seed: int = 42,
    save_best_yaml: bool = True,
    create_plots: bool = True,
) -> tuple[optuna.Study, dict[str, Any]]:
    """
    Run hyperparameter tuning for a specified model.

    Args:
        model_type: Type of model to tune
        gauge_ids: List of gauge IDs for training
        datamodule_config: Configuration for data module
        training_config: Configuration for training
        output_dir_study: Directory to save study results
        experiment_name: Name of the experiment/study
        n_trials: Number of trials to run
        search_spaces_dir: Directory containing search space definitions
        optuna_storage_url: URL for storing optuna results (default: SQLite in output_dir)
        optuna_direction: Direction of optimization ('minimize' or 'maximize')
        optuna_sampler: Custom optuna sampler
        metric_to_optimize: Metric to optimize
        seed: Random seed
        save_best_yaml: Whether to save best parameters to YAML
        create_plots: Whether to create visualization plots

    Returns:
        tuple containing:
        - optuna.Study: The completed study
        - dict[str, Any]: Best hyperparameters configuration

    Raises:
        ConfigurationError: If configuration setup fails
        FileOperationError: If file operations fail
        ModelTrainingError: If model training fails
    """
    from .seed_manager import SeedManager
    seed_manager = SeedManager(seed)
    seed_manager.set_global_seeds()
    output_dir_study_path = Path(output_dir_study)
    study_dir = output_dir_study_path / experiment_name
    study_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint and log directories for this study
    study_checkpoint_dir = study_dir / checkpoint_manager.CHECKPOINTS_DIR_NAME / model_type
    study_log_dir = study_dir / checkpoint_manager.LOGS_DIR_NAME / model_type
    study_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    study_log_dir.mkdir(parents=True, exist_ok=True)

    if optuna_storage_url is None:
        optuna_storage_url = f"sqlite:///{study_dir}/{experiment_name}_study.db"

    logger.info(f"Optuna study storage: {optuna_storage_url}, Name: {experiment_name}")

    try:
        search_space = _load_search_space(model_type, search_spaces_dir)
    except (ConfigurationError, FileOperationError) as e:
        logger.error(f"Failed to load search space for {model_type}: {e}")
        raise ConfigurationError(f"Failed to load search space: {e}")

    def _objective(trial: optuna.Trial) -> float:
        model_pl: pl.LightningModule | None = None
        datamodule: HydroInMemoryDataModule | None = None
        trainer_pl: pl.Trainer | None = None

        try:
            # Create directories for this trial using the checkpoint_manager
            checkpoint_trial_attempt_path_result = checkpoint_manager.determine_output_run_attempt_path(
                base_model_output_dir=study_checkpoint_dir,
                run_index=trial.number,  # Use trial number as run index
                override_previous_attempts=False,
            )

            checkpoint_dir_for_trial = checkpoint_trial_attempt_path_result

            log_trial_attempt_path_result = checkpoint_manager.determine_log_run_attempt_path(
                base_model_log_dir=study_log_dir,
                run_index=trial.number,  # Use trial number as run index
                override_previous_attempts=False,
            )

            log_dir_for_trial = log_trial_attempt_path_result

            # Store path information in trial attributes
            trial.set_user_attr("checkpoint_dir", str(checkpoint_dir_for_trial))
            trial.set_user_attr("log_dir", str(log_dir_for_trial))

            # 1. Suggest hyperparameters for this trial
            trial_hparams = _suggest_trial_hparams(trial, search_space)
            trial.set_user_attr("hparams", trial_hparams)
            logger.info(f"Trial {trial.number} suggested HPs: {trial_hparams}")

            # 2. Configure datamodule with trial hyperparameters using the core helper
            # trial_hparams will provide overrides for input_length, batch_size etc.
            try:
                datamodule = _setup_datamodule_core(
                    base_datamodule_config=datamodule_config,
                    hps_for_datamodule=trial_hparams,
                    gauge_ids=gauge_ids,
                    model_type=model_type,
                )
            except Exception as e:
                return _handle_trial_error(f"DataModule setup error: {e}", trial)

            # 3. Finalize model HPs using the configured datamodule
            finalized_model_hps = _finalize_model_hyperparameters(
                model_hps=trial_hparams, datamodule=datamodule, model_type=model_type
            )
            trial.set_user_attr("model_config", finalized_model_hps)
            logger.info(f"Trial {trial.number} finalized model config: {finalized_model_hps}")

            # 4. Instantiate model with the finalized configuration
            try:
                model_pl, _ = model_factory.create_model_from_config_dict(
                    model_type=model_type, config_dict=finalized_model_hps
                )
            except Exception as e:
                return _handle_trial_error(f"Model instantiation error: {e}", trial)

            # 5. Configure callbacks consistently with other training functions
            callbacks_config_hpt = {
                "with_early_stopping": True,
                "with_model_checkpoint": False,  # Explicitly disable checkpointing for HPT
                "with_lr_monitor": True,
                "with_tensorboard_logger": True,
            }

            try:
                trainer_pl = _configure_trainer_core(
                    training_config=training_config,
                    callbacks_config=callbacks_config_hpt,
                    is_hpt_trial=True,
                    hpt_metric_to_monitor=metric_to_optimize,
                    optuna_trial_for_pruning=None,
                    checkpoint_dir_for_run=checkpoint_dir_for_trial,
                    log_dir_for_run=log_dir_for_trial,
                    model_type_for_paths=model_type,
                    run_idx_for_paths=trial.number,
                )
            except Exception as e:
                return _handle_trial_error(f"Trainer setup error: {e}", trial)

            # 6. Train the model
            trainer_pl.fit(model_pl, datamodule=datamodule)

            # 7. Extract and return performance metric
            metric_val_tensor = trainer_pl.callback_metrics.get(metric_to_optimize)
            if metric_val_tensor is None:
                logger.warning(f"Trial {trial.number}: Metric '{metric_to_optimize}' not found. Pruning.")
                raise optuna.exceptions.TrialPruned(f"Metric '{metric_to_optimize}' not found.")
            final_value = metric_val_tensor.item()
            trial.set_user_attr(metric_to_optimize, final_value)
            logger.info(f"Trial {trial.number} finished. {metric_to_optimize}: {final_value}")
            return final_value

        except optuna.exceptions.TrialPruned as e:
            logger.info(f"Trial {trial.number} pruned: {e}")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number}: Unhandled error: {e}", exc_info=True)
            return _handle_trial_error(f"Unhandled error: {e}", trial)
        finally:
            if trainer_pl and hasattr(trainer_pl, "logger") and trainer_pl.logger:
                try:
                    trainer_pl.logger.finalize("completed")  # type: ignore
                except Exception as e:
                    logger.warning(f"Error finalizing logger for trial {trial.number}: {e}")

            del model_pl, datamodule, trainer_pl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    study = optuna.create_study(
        study_name=experiment_name,
        storage=optuna_storage_url,
        direction=optuna_direction,
        sampler=optuna_sampler,
        load_if_exists=True,
    )
    study.set_user_attr("model_type", model_type)

    best_config: dict[str, Any] = {}

    try:
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=training_config.get("optuna_timeout_per_study_seconds"),
            gc_after_trial=training_config.get("optuna_gc_after_trial", True),
        )

        # Analyze HPT results after study optimization
        best_config = analyze_hpt_results(
            study=study,
            output_dir=study_dir,
            model_type=model_type,
            save_yaml=save_best_yaml,
            create_plots=create_plots,
        )

    except KeyboardInterrupt:
        logger.warning(f"Optuna optimization for study '{experiment_name}' interrupted.")
    except (ConfigurationError, FileOperationError, ModelTrainingError) as e:
        logger.error(f"Error during study optimization for '{experiment_name}': {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during study.optimize for '{experiment_name}': {e}", exc_info=True)
        raise ModelTrainingError(f"Study optimization failed: {e}")
    finally:
        logger.info(f"Optuna study '{experiment_name}' process finished or interrupted.")

    return study, best_config
