import gc
import importlib
import logging
from pathlib import Path
from typing import Any

import optuna
import pytorch_lightning as pl
import torch
from returns.result import Failure, Result, Success

from hydro_forecasting.data.in_memory_datamodule import HydroInMemoryDataModule
from hydro_forecasting.models import model_factory

from .training_runner import _configure_trainer_core, _finalize_model_hyperparameters, _setup_datamodule_core

logger = logging.getLogger(__name__)


def _handle_trial_error(
    error_message: str,
    trial: optuna.Trial,
) -> float:
    logger.error(f"Trial {trial.number}: {error_message}")
    raise optuna.exceptions.TrialPruned(error_message)


def _load_search_space(model_type: str, search_spaces_dir: str | Path = "search_spaces") -> Result[dict[str, Any], str]:
    search_spaces_path = Path(search_spaces_dir)
    model_space_file = search_spaces_path / f"{model_type.lower()}_space.py"
    if not model_space_file.exists():
        return Failure(f"Search space file not found: {model_space_file}")
    try:
        spec = importlib.util.spec_from_file_location(f"search_spaces.{model_type.lower()}_space", model_space_file)
        if spec is None or spec.loader is None:
            return Failure(f"Could not create module spec for {model_space_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if not hasattr(module, "get_search_space"):
            return Failure(f"'get_search_space' function not found in {model_space_file}")
        return Success(module.get_search_space())
    except ImportError as e:
        return Failure(f"Error importing search space for {model_type}: {e}")
    except Exception as e:
        return Failure(f"Unexpected error loading search space for {model_type}: {e}")


def _suggest_trial_hparams(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
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
    pl.seed_everything(seed, workers=True)
    output_dir_study_path = Path(output_dir_study)
    study_dir = output_dir_study_path / experiment_name
    study_dir.mkdir(parents=True, exist_ok=True)

    if optuna_storage_url is None:
        optuna_storage_url = f"sqlite:///{study_dir}/{experiment_name}_study.db"

    logger.info(f"Optuna study storage: {optuna_storage_url}, Name: {experiment_name}")

    search_space_result = _load_search_space(model_type, search_spaces_dir)
    if not isinstance(search_space_result, Success):  # Check for Failure
        err_msg = search_space_result.failure()
        logger.error(f"Failed to load search space for {model_type}: {err_msg}")
        raise ValueError(f"Failed to load search space: {err_msg}")
    search_space = search_space_result.unwrap()

    def _objective(trial: optuna.Trial) -> float:
        model_pl: pl.LightningModule | None = None
        datamodule: HydroInMemoryDataModule | None = None
        trainer_pl: pl.Trainer | None = None

        try:
            # 1. Suggest hyperparameters for this trial
            trial_hparams = _suggest_trial_hparams(trial, search_space)
            trial.set_user_attr("hparams", trial_hparams)
            logger.info(f"Trial {trial.number} suggested HPs: {trial_hparams}")

            # 2. Configure datamodule with trial hyperparameters using the core helper
            # trial_hparams will provide overrides for input_length, batch_size etc.
            datamodule_result = _setup_datamodule_core(
                base_datamodule_config=datamodule_config,
                hps_for_datamodule=trial_hparams,  # Pass all trial HPs
                gauge_ids=gauge_ids,
                model_type=model_type,
            )
            if not isinstance(datamodule_result, Success):
                return _handle_trial_error(f"DataModule config error: {datamodule_result.failure()}", trial)
            datamodule = datamodule_result.unwrap()

            # 3. Finalize model HPs using the configured datamodule
            finalized_model_hps = _finalize_model_hyperparameters(
                model_hps=trial_hparams, datamodule=datamodule, model_type=model_type
            )
            trial.set_user_attr("model_config", finalized_model_hps)
            logger.info(f"Trial {trial.number} finalized model config: {finalized_model_hps}")

            try:
                model_pl, _ = model_factory.create_model_from_config_dict(
                    model_type=model_type, config_dict=finalized_model_hps
                )
            except Exception as e:
                return _handle_trial_error(f"Model instantiation error: {e}", trial)

            # 5. Configure trainer using core helper
            callbacks_config_hpt = {
                "with_early_stopping": True,
                "with_model_checkpoint": False,
                "with_lr_monitor": False,
                "with_tensorboard_logger": False,
                "with_optuna_pruning": True,
            }
            trainer_result = _configure_trainer_core(
                training_config=training_config,  # Base training_config
                callbacks_config=callbacks_config_hpt,
                is_hpt_trial=True,
                hpt_metric_to_monitor=metric_to_optimize,
                optuna_trial_for_pruning=trial,
                # Path arguments are not needed for HPT trials
            )
            if not isinstance(trainer_result, Success):
                return _handle_trial_error(f"Trainer config error: {trainer_result.failure()}", trial)
            trainer_pl = trainer_result.unwrap()

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
            del model_pl, datamodule, trainer_pl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    study = optuna.create_study(
        study_name=experiment_name,
        storage=optuna_storage_url,
        direction=optuna_direction,
        sampler=optuna_sampler,
        pruner=optuna_pruner,
        load_if_exists=True,
    )
    study.set_user_attr("model_type", model_type)

    try:
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=training_config.get("optuna_timeout_per_study_seconds"),
            gc_after_trial=training_config.get("optuna_gc_after_trial", True),
        )
    except KeyboardInterrupt:
        logger.warning(f"Optuna optimization for study '{experiment_name}' interrupted.")
    except Exception as e:
        logger.error(f"Unexpected error during study.optimize for '{experiment_name}': {e}", exc_info=True)
    finally:
        logger.info(f"Optuna study '{experiment_name}' process finished or interrupted.")
    return study
