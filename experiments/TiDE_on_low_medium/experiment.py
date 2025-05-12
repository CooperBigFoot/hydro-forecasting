import argparse
import dataclasses
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from returns.result import Result

from .config import ExperimentConfig
from .data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


from hydro_forecasting.data.in_memory_datamodule import HydroInMemoryDataModule
from hydro_forecasting.experiment_utils import checkpoint_manager
from hydro_forecasting.experiment_utils.checkpoint_manager import CHECKPOINTS_DIR_NAME, LOGS_DIR_NAME
from hydro_forecasting.model_evaluation.hp_from_yaml import hp_from_yaml
from hydro_forecasting.models import model_factory

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global random seed set to: {seed}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hydrological Forecasting Experiment")

    parser.add_argument("--experiment_config_name", type=str, default="config")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_types", type=str, nargs="+", required=True)
    parser.add_argument("--yaml_dir", type=str, default="./yaml_files/")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--override_previous_attempts", action="store_true")
    parser.add_argument("--load_checkpoint_from_dir", type=str, default=None)
    parser.add_argument("--select_overall_best_checkpoint", action="store_true")
    parser.add_argument("--choose_run_checkpoint_idx", type=int, default=None)
    parser.add_argument("--use_attempt_checkpoint_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, help="Override ExperimentConfig.batch_size")
    parser.add_argument("--max_epochs", type=int, help="Override ExperimentConfig.max_epochs")
    parser.add_argument("--learning_rate_override", type=float, help="Override learning_rate from YAML.")

    args = parser.parse_args()
    if (
        args.load_checkpoint_from_dir
        and not args.select_overall_best_checkpoint
        and args.choose_run_checkpoint_idx is None
    ):
        parser.error(
            "--choose_run_checkpoint_idx is required if --load_checkpoint_from_dir is set and not --select_overall_best_checkpoint."
        )
    return args


def main():
    args = parse_arguments()
    logger.info(f"Parsed arguments: {args}")

    set_random_seeds(args.base_seed)

    exp_config_base = ExperimentConfig()

    exp_config_dict = dataclasses.asdict(exp_config_base)
    cli_overrides_applied_count = 0
    for field_name, field_value in vars(args).items():
        if field_value is not None and field_name in exp_config_dict:
            if isinstance(exp_config_dict[field_name], Path):
                exp_config_dict[field_name] = Path(field_value) if field_value else None
            else:
                exp_config_dict[field_name] = field_value
            logger.info(f"Overriding ExperimentConfig.{field_name} with CLI value: {field_value}")
            cli_overrides_applied_count += 1

    current_exp_config = ExperimentConfig(**exp_config_dict) if cli_overrides_applied_count > 0 else exp_config_base

    if args.output_dir:
        current_exp_config.base_output_dir = Path(args.output_dir)
    current_exp_config.base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using base output directory: {current_exp_config.base_output_dir}")

    if not current_exp_config.path_to_preprocessing_output_directory.is_absolute():
        current_exp_config.path_to_preprocessing_output_directory = (
            current_exp_config.base_output_dir / current_exp_config.path_to_preprocessing_output_directory
        )
    current_exp_config.path_to_preprocessing_output_directory.mkdir(parents=True, exist_ok=True)

    yaml_dir_path = Path(args.yaml_dir)
    if not yaml_dir_path.is_absolute():
        yaml_dir_path = (Path(__file__).parent / yaml_dir_path).resolve()

    for model_type in args.model_types:
        logger.info(f"===== Processing Model Type: {model_type.upper()} =====")
        model_type_results: list[dict[str, Any]] = []

        model_yaml_path = yaml_dir_path / f"{model_type.lower()}.yaml"
        if not model_yaml_path.exists():
            logger.error(f"YAML for model '{model_type}' not found at: {model_yaml_path}. Skipping.")
            continue

        model_checkpoints_base_dir = current_exp_config.base_output_dir / CHECKPOINTS_DIR_NAME / model_type
        model_logs_base_dir = current_exp_config.base_output_dir / LOGS_DIR_NAME / model_type
        model_checkpoints_base_dir.mkdir(parents=True, exist_ok=True)
        model_logs_base_dir.mkdir(parents=True, exist_ok=True)

        try:
            base_model_hp = hp_from_yaml(model_type, str(model_yaml_path))
        except Exception as e:
            logger.error(f"Could not load HPs for {model_type} from {model_yaml_path}: {e}. Skipping.")
            continue

        for run_idx in range(args.num_runs):
            current_run_seed = args.base_seed + run_idx
            set_random_seeds(current_run_seed)
            logger.info(f"--- Run: {run_idx + 1}/{args.num_runs} for {model_type} (Seed: {current_run_seed}) ---")

            checkpoint_run_attempt_path_result: Result[Path, str] = (
                checkpoint_manager.determine_output_run_attempt_path(
                    base_model_output_dir=model_checkpoints_base_dir,
                    run_index=run_idx,
                    override_previous_attempts=args.override_previous_attempts,
                )
            )
            if not isinstance(checkpoint_run_attempt_path_result, Result):
                logger.error(
                    f"Checkpoint path error for {model_type}/run_{run_idx}: {checkpoint_run_attempt_path_result.failure()}. Skip run."
                )
                continue
            current_checkpoint_attempt_dir = checkpoint_run_attempt_path_result.unwrap()

            log_run_attempt_path_result: Result[Path, str] = checkpoint_manager.determine_log_run_attempt_path(
                base_model_log_dir=model_logs_base_dir,
                run_index=run_idx,
                override_previous_attempts=args.override_previous_attempts,
            )
            if not isinstance(log_run_attempt_path_result, Result):
                logger.error(
                    f"Log path error for {model_type}/run_{run_idx}: {log_run_attempt_path_result.failure()}. Skip run."
                )
                continue
            current_log_attempt_dir = log_run_attempt_path_result.unwrap()

            model_instance: pl.LightningModule | None = None
            final_model_hp = base_model_hp.copy()

            try:
                model_instance, created_hps = model_factory.create_model(model_type, str(model_yaml_path))
                final_model_hp.update(created_hps)
            except Exception as e:
                logger.error(f"Error creating new model: {e}", exc_info=True)
                continue

            if model_instance is None:
                logger.error("Model instance is None. Skip run.")
                continue

            current_exp_config.input_length = final_model_hp.get("input_len")
            current_exp_config.output_length = final_model_hp.get("output_len")
            if current_exp_config.input_length is None or current_exp_config.output_length is None:
                logger.error(f"input_len/output_len not in HPs for {model_type}. Skip run.")
                continue

            try:
                list_of_gauge_ids = load_data(current_exp_config, PROJECT_ROOT, **vars(args))
                if not list_of_gauge_ids:
                    logger.warning("No gauge IDs from data_loader. Skip run.")
                    continue

                datamodule = HydroInMemoryDataModule(
                    region_time_series_base_dirs=current_exp_config.region_time_series_base_dirs,
                    region_static_attributes_base_dirs=current_exp_config.region_static_attributes_base_dirs,
                    path_to_preprocessing_output_directory=current_exp_config.path_to_preprocessing_output_directory,
                    group_identifier=current_exp_config.group_identifier,
                    batch_size=current_exp_config.batch_size,
                    input_length=current_exp_config.input_length,
                    output_length=current_exp_config.output_length,
                    forcing_features=current_exp_config.forcing_features,
                    static_features=current_exp_config.static_features,
                    target=current_exp_config.target,
                    preprocessing_configs=current_exp_config.preprocessing_configs,
                    num_workers=current_exp_config.num_workers,
                    min_train_years=current_exp_config.min_train_years,
                    train_prop=current_exp_config.train_prop,
                    val_prop=current_exp_config.val_prop,
                    test_prop=current_exp_config.test_prop,
                    max_imputation_gap_size=current_exp_config.max_imputation_gap_size,
                    chunk_size=current_exp_config.chunk_size,
                    validation_chunk_size=current_exp_config.validation_chunk_size,
                    list_of_gauge_ids_to_process=list_of_gauge_ids,
                    domain_id=current_exp_config.domain_id,
                    domain_type=current_exp_config.domain_type,
                    is_autoregressive=current_exp_config.is_autoregressive,
                )
                datamodule.prepare_data()
                datamodule.setup(stage="fit")
            except Exception as e:
                logger.error(f"Error setting up DataModule: {e}", exc_info=True)
                continue

            early_stopping_cb = EarlyStopping(
                monitor="val_loss", patience=current_exp_config.early_stopping_patience, verbose=True, mode="min"
            )
            checkpoint_filename = f"{model_type}-run{run_idx}-{{epoch:02d}}-{{val_loss:.4f}}"
            model_checkpoint_cb = ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(current_checkpoint_attempt_dir),
                filename=checkpoint_filename,
                save_top_k=current_exp_config.save_top_k_checkpoints,
                mode="min",
                save_last=True,
            )
            lr_monitor_cb = LearningRateMonitor(logging_interval="step")
            tb_logger = TensorBoardLogger(
                save_dir=str(current_log_attempt_dir.parent),
                name=current_log_attempt_dir.name,
                version="",
                default_hp_metric=False,
            )
            trainer = pl.Trainer(
                accelerator=current_exp_config.accelerator,
                devices=current_exp_config.devices,
                max_epochs=current_exp_config.max_epochs,
                enable_progress_bar=True,
                logger=tb_logger,
                callbacks=[early_stopping_cb, model_checkpoint_cb, lr_monitor_cb],
                num_sanity_val_steps=current_exp_config.num_sanity_val_steps,
                reload_dataloaders_every_n_epochs=current_exp_config.reload_dataloaders_every_n_epochs,
            )
            torch.set_float32_matmul_precision(current_exp_config.torch_float32_matmul_precision)

            best_model_path_for_attempt: str | None = None  # Initialize
            try:
                logger.info(f"Training {model_type}, Run {run_idx + 1}, Attempt: {current_checkpoint_attempt_dir.name}")
                trainer.fit(model_instance, datamodule=datamodule)
                best_model_path_for_attempt = model_checkpoint_cb.best_model_path
                best_model_score_for_attempt = model_checkpoint_cb.best_model_score
                if best_model_score_for_attempt is not None:
                    best_model_score_for_attempt = best_model_score_for_attempt.item()

                if best_model_path_for_attempt:
                    relative_best_path = Path(best_model_path_for_attempt).relative_to(model_checkpoints_base_dir)
                    model_type_results.append(
                        {
                            "checkpoint_path_absolute": best_model_path_for_attempt,
                            "checkpoint_path_relative": str(relative_best_path),
                            "val_loss": best_model_score_for_attempt,
                            "run_index": run_idx,
                            "attempt_dir_name": current_checkpoint_attempt_dir.name,
                            "seed": current_run_seed,
                        }
                    )
                else:
                    logger.warning(f"No best model path for attempt (Run {run_idx + 1}).")
            except Exception as e:
                logger.error(f"Error training {model_type}, Run {run_idx + 1}: {e}", exc_info=True)
            finally:
                if tb_logger:
                    tb_logger.finalize("success" if best_model_path_for_attempt else "failed")

        if model_type_results:
            sorted_results = sorted(
                model_type_results, key=lambda x: x["val_loss"] if x["val_loss"] is not None else float("inf")
            )
            overall_best_run_for_model_type = sorted_results[0]
            logger.info(f"Overall best for '{model_type}' this exec: {overall_best_run_for_model_type}")
            update_result = checkpoint_manager.update_overall_best_model_info_file(
                model_checkpoints_output_dir=model_checkpoints_base_dir,
                best_checkpoint_relative_path_within_model_dir=overall_best_run_for_model_type[
                    "checkpoint_path_relative"
                ],
            )
            if not isinstance(update_result, Result):
                logger.error(f"Failed to update best model info for {model_type}: {update_result.failure()}")
        else:
            logger.warning(f"No successful runs for '{model_type}'. Cannot determine overall best.")

    logger.info("===== Experiment Script Finished =====")


if __name__ == "__main__":
    main()
