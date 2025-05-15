import copy
import gc
import importlib
import logging
from pathlib import Path
from typing import Any

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from returns.result import Failure, Result, Success, safe

from hydro_forecasting.data.in_memory_datamodule import HydroInMemoryDataModule
from hydro_forecasting.models import model_factory

logger = logging.getLogger(__name__)


def _handle_trial_error(
    error_message: str,
    trial: optuna.Trial,
) -> float:
    """
    Handles errors within an Optuna trial by logging and pruning.

    Args:
        error_message: The error message to log.
        trial: The Optuna trial object.

    Returns:
        This function will always raise TrialPruned and not return a float directly.
        The float return type is for signature compatibility if an alternative
        error handling (like returning float('inf')) was used.
    """
    logger.error(f"Trial {trial.number}: {error_message}")
    raise optuna.exceptions.TrialPruned(error_message)


def _load_search_space(model_type: str, search_spaces_dir: str | Path = "search_spaces") -> Result[dict[str, Any], str]:
    """
    Loads the hyperparameter search space for a given model type.

    The search space is expected to be defined in a Python file named
    `[model_type]_space.py` within the `search_spaces_dir`. This file
    should contain a function `get_search_space()` that returns the
    search space dictionary.

    Args:
        model_type: The type of the model (e.g., "tsmixer", "tide").
        search_spaces_dir: The directory containing the search space files.

    Returns:
        Result[Dict[str, Any], str]: A dictionary defining the search space
                                     or an error message.
    """
    search_spaces_path = Path(search_spaces_dir)
    model_space_file = search_spaces_path / f"{model_type.lower()}_space.py"

    if not model_space_file.exists():
        return Failure(f"Search space file not found: {model_space_file}")

    try:
        spec = importlib.util.spec_from_file_location(f"search_spaces.{model_type.lower()}_space", model_space_file)
        if spec is None or spec.loader is None:
            return Failure(f"Could not create module spec for {model_space_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "get_search_space"):
            return Failure(f"'get_search_space' function not found in {model_space_file}")

        return Success(module.get_search_space())
    except ImportError as e:
        return Failure(f"Error importing search space for {model_type}: {e}")
    except Exception as e:
        return Failure(f"Unexpected error loading search space for {model_type}: {e}")


def _suggest_trial_hparams(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """
    Suggests hyperparameters for the current Optuna trial based on the search space.

    Args:
        trial: The Optuna trial object.
        search_space: The search space dictionary loaded for the model.

    Returns:
        A dictionary of suggested hyperparameters.
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
                param_name,
                config["low"],
                config["high"],
                step=config.get("step", 1),
                log=config.get("log", False),
            )
        elif param_type == "float":
            hparams[param_name] = trial.suggest_float(
                param_name,
                config["low"],
                config["high"],
                step=config.get("step"),
                log=config.get("log", False),
            )
    return hparams


def _map_hparams_to_model_config(
    model_type: str,
    trial_hparams: dict[str, Any],
    datamodule: HydroInMemoryDataModule,
) -> dict[str, Any]:
    """
    Maps trial hyperparameters to a model configuration dictionary, adding
    necessary parameters derived from the datamodule.

    Args:
        model_type: The type of the model.
        trial_hparams: Hyperparameters suggested by Optuna for this trial.
        datamodule: The configured datamodule to extract dimensions from.

    Returns:
        A dictionary formatted for instantiating the model's configuration class.
    """
    # Start with trial hyperparameters
    model_config_dict = copy.deepcopy(trial_hparams)

    if "input_length" in model_config_dict:
        model_config_dict["input_len"] = model_config_dict.pop("input_length")

    model_config_dict["input_size"] = len(datamodule.input_features_ordered_for_X)
    model_config_dict["static_size"] = len(datamodule.static_features) if datamodule.static_features else 0
    model_config_dict["future_input_size"] = len(datamodule.future_features_ordered)
    model_config_dict["group_identifier"] = datamodule.group_identifier
    model_config_dict["output_len"] = datamodule.output_length

    return model_config_dict


@safe
def _configure_datamodule_for_trial(
    base_datamodule_config: dict[str, Any],
    trial_hparams: dict[str, Any],
    gauge_ids: list[str],
) -> HydroInMemoryDataModule:
    """
    Configures and sets up the HydroInMemoryDataModule for the current trial.

    Args:
        base_datamodule_config: The base configuration for the datamodule.
        trial_hparams: Hyperparameters suggested by Optuna for this trial.
        gauge_ids: List of gauge IDs for training and validation.

    Returns:
        Result[HydroInMemoryDataModule, str]: The configured datamodule or an
                                             error message.
    """
    current_datamodule_config = copy.deepcopy(base_datamodule_config)

    # Apply trial hyperparameters to datamodule config. This should be handled differently tho :()
    if "input_length" in trial_hparams:
        current_datamodule_config["input_length"] = trial_hparams["input_length"]
    elif "input_len" in trial_hparams:
        current_datamodule_config["input_length"] = trial_hparams["input_len"]

    datamodule = HydroInMemoryDataModule(list_of_gauge_ids_to_process=gauge_ids, **current_datamodule_config)
    datamodule.prepare_data()
    datamodule.setup()

    logger.info(
        f"Trial datamodule configured. Input length: {datamodule.input_length}, "
        f"Output length: {datamodule.output_length}, "
        f"Batch size: {datamodule.batch_size}"
    )

    return datamodule


@safe
def _instantiate_model_for_trial(
    model_type: str,
    mapped_model_config: dict[str, Any],
) -> pl.LightningModule:
    """
    Instantiates the model for the current trial using the mapped model configuration.

    Args:
        model_type: The type of model to instantiate.
        mapped_model_config: The fully prepared configuration dictionary for the model.

    Returns:
        Result[pl.LightningModule, str]: The instantiated model or an error message.
    """
    model, _ = model_factory.create_model_from_config_dict(
        model_type=model_type,
        config_dict=mapped_model_config,
    )
    return model


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
    optuna_pruner: optuna.pruners.BasePruner | None = None,
    metric_to_optimize: str = "val_loss",
    seed: int = 42,
) -> optuna.Study:
    """
    Performs hyperparameter optimization for a given model type using Optuna.

    Args:
        model_type: The type of model to tune (e.g., "ealstm", "tsmixer").
        gauge_ids: List of basin identifiers for training/validation.
        datamodule_config: Configuration dictionary for HydroInMemoryDataModule.
        training_config: Configuration dictionary for PyTorch Lightning Trainer.
        output_dir_study: Path to save Optuna study artifacts.
        experiment_name: Name for the Optuna study.
        n_trials: Number of optimization trials to run.
        search_spaces_dir: Directory containing model-specific search space files.
        optuna_storage_url: Optional URL for Optuna's storage (e.g., SQLite).
                            If None, a local SQLite DB is created in output_dir_study.
        optuna_direction: "minimize" or "maximize" the metric_to_optimize.
        optuna_sampler: Optional Optuna sampler.
        optuna_pruner: Optional Optuna pruner.
        metric_to_optimize: The metric to monitor and optimize (e.g., "val_loss").
        seed: Global random seed for reproducibility.

    Returns:
        The Optuna Study object containing results of the optimization.
    """
    pl.seed_everything(seed, workers=True)
    output_dir_study_path = Path(output_dir_study)
    study_dir = output_dir_study_path / experiment_name
    study_dir.mkdir(parents=True, exist_ok=True)

    if optuna_storage_url is None:
        optuna_storage_url = f"sqlite:///{study_dir}/{experiment_name}_study.db"

    logger.info(f"Optuna study storage: {optuna_storage_url}")
    logger.info(f"Optuna study name: {experiment_name}")
    logger.info(f"Optimization direction: {optuna_direction}")
    logger.info(f"Metric to optimize: {metric_to_optimize}")
    logger.info(f"Number of trials: {n_trials}")

    search_space_result = _load_search_space(model_type, search_spaces_dir)
    if isinstance(search_space_result, Failure):
        err_msg = search_space_result.failure()
        logger.error(f"Failed to load search space for {model_type}: {err_msg}")
        raise ValueError(f"Failed to load search space: {err_msg}")
    search_space = search_space_result.unwrap()

    logger.info(f"Loaded search space for {model_type}: {search_space}")

    def _objective(trial: optuna.Trial) -> float:
        """
        Objective function for an Optuna trial. Trains and evaluates a model.
        """
        model_pl: pl.LightningModule | None = None
        datamodule: HydroInMemoryDataModule | None = None
        trainer: pl.Trainer | None = None

        try:
            # 1. Suggest hyperparameters for this trial
            trial_hparams = _suggest_trial_hparams(trial, search_space)
            logger.info(f"Trial {trial.number}: Suggested HPs: {trial_hparams}")
            trial.set_user_attr("hparams", trial_hparams)

            # 2. Configure datamodule with trial hyperparameters
            datamodule_result = _configure_datamodule_for_trial(datamodule_config, trial_hparams, gauge_ids)
            if isinstance(datamodule_result, Failure):
                err = datamodule_result.failure()
                return _handle_trial_error(f"DataModule config error: {err}", trial)

            datamodule = datamodule_result.unwrap()

            # 3. Map hyperparameters to model configuration format
            mapped_model_config = _map_hparams_to_model_config(model_type, trial_hparams, datamodule)
            logger.info(f"Trial {trial.number}: Mapped Model Config: {mapped_model_config}")
            trial.set_user_attr("model_config", mapped_model_config)

            # 4. Instantiate model with the configuration
            model_result = _instantiate_model_for_trial(model_type, mapped_model_config)
            if isinstance(model_result, Failure):
                err = model_result.failure()
                return _handle_trial_error(f"Model instantiation error: {err}", trial)

            model_pl = model_result.unwrap()

            # 5. Configure training callbacks
            callbacks = []
            early_stopping_config = training_config.get("early_stopping", {})
            if early_stopping_config.get("enabled", True):
                monitor = early_stopping_config.get("monitor", metric_to_optimize)
                patience = early_stopping_config.get("patience", 5)
                es_mode = early_stopping_config.get("mode", "min" if "loss" in metric_to_optimize else "max")
                callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode=es_mode, verbose=False))
                logger.info(
                    f"Trial {trial.number}: EarlyStopping enabled (monitor='{monitor}', "
                    f"patience={patience}, mode='{es_mode}')."
                )

            # 6. Configure and create trainer
            trainer_params = {
                "max_epochs": training_config.get("max_epochs_per_trial", training_config.get("max_epochs", 10)),
                "accelerator": training_config.get("accelerator", "auto"),
                "devices": training_config.get("devices", "auto"),
                "callbacks": callbacks,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": training_config.get("enable_progress_bar_in_trial", False),
                "num_sanity_val_steps": training_config.get("num_sanity_val_steps", 0),
                "deterministic": training_config.get("deterministic", True),
                "gradient_clip_val": training_config.get("gradient_clip_val"),
                **training_config.get("additional_trainer_args_for_trial", {}),
            }
            trainer = pl.Trainer(**trainer_params)

            # 7. Train the model
            trainer.fit(model_pl, datamodule=datamodule)

            # 8. Extract and return performance metric
            metric_val_tensor = trainer.callback_metrics.get(metric_to_optimize)
            if metric_val_tensor is None:
                logger.warning(
                    f"Trial {trial.number}: Metric '{metric_to_optimize}' not "
                    "found in callback_metrics. Available metrics: "
                    f"{list(trainer.callback_metrics.keys())}. Pruning trial."
                )
                raise optuna.exceptions.TrialPruned(f"Metric '{metric_to_optimize}' not found.")

            final_value = metric_val_tensor.item()
            logger.info(f"Trial {trial.number} finished. {metric_to_optimize}: {final_value}")
            trial.set_user_attr(metric_to_optimize, final_value)
            return final_value

        except optuna.exceptions.TrialPruned as e:
            logger.info(f"Trial {trial.number} pruned: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Trial {trial.number}: Unhandled error during objective execution: {e}",
                exc_info=True,
            )
            return _handle_trial_error(f"Unhandled error: {e}", trial)
        finally:
            # Clean up resources
            if "model_pl" in locals() and model_pl is not None:
                del model_pl
            if "datamodule" in locals() and datamodule is not None:
                del datamodule
            if "trainer" in locals() and trainer is not None:
                del trainer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Create and configure the study
    study = optuna.create_study(
        study_name=experiment_name,
        storage=optuna_storage_url,
        direction=optuna_direction,
        sampler=optuna_sampler,
        pruner=optuna_pruner,
        load_if_exists=True,
    )

    # Set study metadata
    study.set_user_attr("model_type", model_type)
    study.set_user_attr("metric_optimized", metric_to_optimize)
    study.set_user_attr("datamodule_config_keys", list(datamodule_config.keys()))
    study.set_user_attr("training_config_keys", list(training_config.keys()))
    study.set_user_attr("gauge_ids_count", len(gauge_ids))
    if gauge_ids:
        study.set_user_attr("gauge_ids_sample", gauge_ids[: min(5, len(gauge_ids))])

    # Run optimization
    try:
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=training_config.get("optuna_timeout_per_study_seconds"),
            gc_after_trial=training_config.get("optuna_gc_after_trial", True),
        )
    except KeyboardInterrupt:
        logger.warning(f"Optuna optimization for study '{experiment_name}' interrupted by user.")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during study.optimize for '{experiment_name}': {e}",
            exc_info=True,
        )
    finally:
        # Log summary of results
        logger.info(f"Optuna optimization process for '{experiment_name}' finished or was interrupted.")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            logger.info(f"Number of COMPLETED trials: {len(completed_trials)}")
            try:
                best_trial = study.best_trial
                logger.info(f"Best trial ({best_trial.number}) value ({metric_to_optimize}): {best_trial.value}")
                logger.info(f"Best hyperparameters: {best_trial.params}")
            except ValueError:
                logger.info(
                    "No trials completed successfully, so no best trial information to display from study.best_trial."
                )
        else:
            logger.info("No trials were completed successfully.")
        logger.info(f"Total number of trials in study (all states): {len(study.trials)}")

    return study
