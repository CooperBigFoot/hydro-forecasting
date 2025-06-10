import copy
import datetime
import random
from typing import Any

import numpy as np
import polars as pl
import pytorch_lightning as pl_trainer
import torch


class TSForecastEvaluator:
    """Evaluator for time series forecasting models with per-basin metrics support, using Polars."""

    def __init__(
        self,
        horizons: list[int],
        models_and_datamodules: dict[str, tuple[pl_trainer.LightningModule, pl_trainer.LightningDataModule]] = None,
        default_datamodule: pl_trainer.LightningDataModule | None = None,
        benchmark_model: str = None,
        trainer_kwargs: dict[str, Any] = None,
        random_seed: int = 42,  # Add explicit random seed
    ):
        self.horizons = sorted(set(horizons))
        self.default_datamodule = default_datamodule
        self.benchmark_model = benchmark_model
        self.trainer_kwargs = trainer_kwargs or {"accelerator": "cpu", "devices": 1}
        self.random_seed = random_seed
        self.results: dict[str, dict[str, Any]] = {}
        self.models: dict[str, pl_trainer.LightningModule] = {}
        self.datamodules: dict[str, pl_trainer.LightningDataModule] = {}

        if models_and_datamodules:
            for name, (model, datamodule) in models_and_datamodules.items():
                self.models[name] = self._deep_copy_model(model)
                self.datamodules[name] = datamodule

    def _set_random_seeds(self, seed: int = None):
        """Set random seeds for reproducibility."""
        if seed is None:
            seed = self.random_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (might slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _deep_copy_model(self, model: pl_trainer.LightningModule) -> pl_trainer.LightningModule:
        """Create a proper deep copy of a PyTorch Lightning model."""
        # Save the model state
        model_state = model.state_dict()
        model_config = getattr(model, "config", None)

        # Create a new instance with the same class and config
        model_class = model.__class__
        if model_config is not None:
            new_model = model_class(model_config)
        else:
            # Fallback to deepcopy if config not available
            new_model = copy.deepcopy(model)

        # Load the saved state
        if model_config is not None:
            new_model.load_state_dict(model_state, strict=False)

        return new_model

    def _prepare_model_for_testing(self, model: pl_trainer.LightningModule) -> pl_trainer.LightningModule:
        """Prepare a fresh model instance for testing."""
        # Create a fresh copy
        test_model = self._deep_copy_model(model)

        # Set to eval mode
        test_model.eval()

        # Disable gradients
        for param in test_model.parameters():
            param.requires_grad = False

        return test_model

    def register_model(
        self,
        name: str,
        model: pl_trainer.LightningModule,
        datamodule: pl_trainer.LightningDataModule | None = None,
    ):
        self.models[name] = self._deep_copy_model(model)
        if datamodule:
            self.datamodules[name] = datamodule
        elif name not in self.datamodules and self.default_datamodule is None:
            print(f"Warning: No datamodule provided for model '{name}' and no default datamodule available")

    def test_models(self, datamodule: pl_trainer.LightningDataModule | None = None) -> dict[str, dict[str, Any]]:
        """Test all registered models with proper isolation."""
        self.results = {}  # Clear previous results

        for name, model in self.models.items():
            print(f"Testing {name}...")
            result = self.test_specific_model(name, datamodule)
            self.results[name] = result

        return self.results

    def test_specific_model(
        self, name: str, datamodule: pl_trainer.LightningDataModule | None = None
    ) -> dict[str, Any]:
        """Test a specific model with proper isolation."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Get model and datamodule
        original_model = self.models[name]
        model_datamodule = datamodule or self.datamodules.get(name, self.default_datamodule)

        if model_datamodule is None:
            raise ValueError(f"No datamodule found for model '{name}'")

        # Prepare a fresh model instance for testing
        test_model = self._prepare_model_for_testing(original_model)

        # Create a fresh trainer instance for each test
        trainer_config = self.trainer_kwargs.copy()
        trainer_config.update(
            {
                "logger": False,  # Disable logging
                "enable_checkpointing": False,  # Disable checkpointing
                "enable_progress_bar": True,  # Keep progress bar
            }
        )
        trainer = pl_trainer.Trainer(**trainer_config)

        # Run the test
        trainer.test(test_model, datamodule=model_datamodule)

        if not hasattr(test_model, "test_results"):
            raise AttributeError(
                f"Model {name} doesn't have test_results attribute. "
                "Ensure your LightningModule stores test outputs in self.test_results."
            )

        # Evaluate results
        df, metrics, basin_metrics = self.evaluate(test_model.test_results, model_datamodule)

        result = {
            "df": df,
            "metrics": metrics,
            "basin_metrics": basin_metrics,
            "datamodule": model_datamodule,
        }

        # Clean up
        del test_model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    def evaluate(
        self,
        test_results: dict[str, Any],
        datamodule: pl_trainer.LightningDataModule | None = None,
    ) -> tuple[
        pl.DataFrame,
        dict[int, dict[str, float]],
        dict[str, dict[int, dict[str, float]]],
    ]:
        df = self._prepare_evaluation_dataframe(test_results, datamodule)
        overall_metrics = self._calculate_overall_metrics(df)
        basin_metrics = self._calculate_basin_metrics(df)
        return df, overall_metrics, basin_metrics

    def _prepare_evaluation_dataframe(
        self,
        test_results: dict[str, Any],
        datamodule: pl_trainer.LightningDataModule | None = None,
    ) -> pl.DataFrame:
        basin_ids_list = test_results.get("basin_ids")
        preds_tensor = test_results.get("predictions")
        obs_tensor = test_results.get("observations")
        # Expecting a list of integer millisecond timestamps or Nones
        input_end_date_ms_list = test_results.get("input_end_date")

        if basin_ids_list is None or preds_tensor is None or obs_tensor is None:
            raise ValueError("Missing required keys in test_results: 'basin_ids', 'predictions', 'observations'")

        preds_np = preds_tensor.cpu().numpy() if hasattr(preds_tensor, "cpu") else np.array(preds_tensor)
        obs_np = obs_tensor.cpu().numpy() if hasattr(obs_tensor, "cpu") else np.array(obs_tensor)
        basin_ids_np = np.array(basin_ids_list)

        if preds_np.shape != obs_np.shape:
            raise ValueError(f"Prediction shape {preds_np.shape} doesn't match observation shape {obs_np.shape}")
        if preds_np.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape {preds_np.shape}, expected 2D array [num_samples, num_horizons]"
            )

        num_samples = preds_np.shape[0]
        horizons_per_sample = preds_np.shape[1]
        current_horizons = self.horizons
        if horizons_per_sample != len(current_horizons):
            print(
                f"Warning: Model output horizons ({horizons_per_sample}) != evaluator horizons ({len(current_horizons)}). Adjusting."
            )
            current_horizons = list(range(1, horizons_per_sample + 1))

        preds_flat = preds_np.flatten()
        obs_flat = obs_np.flatten()
        if len(basin_ids_np) != num_samples:
            raise ValueError(f"Basin IDs length ({len(basin_ids_np)}) != num samples ({num_samples})")

        basin_ids_expanded = np.repeat(basin_ids_np, horizons_per_sample)
        horizons_expanded = np.tile(np.array(current_horizons), num_samples)

        dates_expanded = []
        if input_end_date_ms_list:
            if len(input_end_date_ms_list) != num_samples:
                print(
                    f"Warning: input_end_date_ms_list length ({len(input_end_date_ms_list)}) != num samples ({num_samples}). Aligning."
                )
                if not input_end_date_ms_list:
                    input_end_date_ms_list = [None] * num_samples  # Default to None if empty
                elif len(input_end_date_ms_list) < num_samples:
                    input_end_date_ms_list.extend(
                        [input_end_date_ms_list[-1]] * (num_samples - len(input_end_date_ms_list))
                    )
                else:
                    input_end_date_ms_list = input_end_date_ms_list[:num_samples]

            base_python_datetimes = []
            for ms_timestamp in input_end_date_ms_list:
                # Handle tensor conversion to CPU if needed
                if hasattr(ms_timestamp, "cpu"):
                    ms_timestamp = (
                        ms_timestamp.cpu().item() if ms_timestamp.numel() == 1 else ms_timestamp.cpu().numpy()
                    )

                if ms_timestamp is None or (isinstance(ms_timestamp, int | float) and np.isnan(ms_timestamp)):
                    base_python_datetimes.append(None)
                else:
                    try:
                        # Convert ms to seconds for fromtimestamp, ensure UTC
                        dt_obj = datetime.datetime.fromtimestamp(int(ms_timestamp) / 1000.0, tz=datetime.timezone.utc)
                        base_python_datetimes.append(dt_obj.replace(tzinfo=None))  # Store as naive for Polars
                    except (ValueError, TypeError, OverflowError) as e:
                        print(f"Warning: Could not convert timestamp {ms_timestamp} to datetime: {e}. Using None.")
                        base_python_datetimes.append(None)

            for base_dt_naive in base_python_datetimes:
                for h_val in current_horizons:
                    if base_dt_naive is None:
                        dates_expanded.append(None)
                    else:
                        dates_expanded.append(base_dt_naive + datetime.timedelta(days=int(h_val)))
        else:
            print("Warning: No 'input_end_date' in test_results. Dates will be missing.")
            dates_expanded = [None] * len(preds_flat)

        schema = {
            "horizon": pl.Int32,
            "prediction": pl.Float32,
            "observed": pl.Float32,
            "basin_id": pl.Utf8,
            "date": pl.Datetime,
        }
        df = pl.DataFrame(
            {
                "horizon": horizons_expanded,
                "prediction": preds_flat,
                "observed": obs_flat,
                "basin_id": basin_ids_expanded,
                "date": dates_expanded,
            },
            schema=schema,
        )

        dm_for_transform = datamodule or self.default_datamodule
        if dm_for_transform and hasattr(dm_for_transform, "inverse_transform_predictions"):
            try:
                pred_numpy = df.get_column("prediction").to_numpy()
                obs_numpy = df.get_column("observed").to_numpy()
                transformed_preds = dm_for_transform.inverse_transform_predictions(pred_numpy, basin_ids_expanded)
                transformed_obs = dm_for_transform.inverse_transform_predictions(obs_numpy, basin_ids_expanded)
                df = df.with_columns(
                    [
                        pl.Series("prediction", transformed_preds, dtype=pl.Float32),
                        pl.Series("observed", transformed_obs, dtype=pl.Float32),
                    ]
                )
            except Exception as e:
                print(f"Warning: Failed to inverse transform predictions: {e}")

        return df

    def _calculate_overall_metrics(self, df: pl.DataFrame) -> dict[int, dict[str, float]]:
        overall_metrics: dict[int, dict[str, float]] = {}
        max_available_horizon = df.get_column("horizon").max() if df.height > 0 else 0

        for h in self.horizons:
            if max_available_horizon is not None and h > max_available_horizon:
                continue
            horizon_data = df.filter(pl.col("horizon") == h)
            if not horizon_data.is_empty():
                overall_metrics[h] = self._calculate_metrics(horizon_data)
            else:
                overall_metrics[h] = dict.fromkeys(["MSE", "MAE", "NSE", "RMSE"], np.nan)
        return overall_metrics

    def _calculate_basin_metrics(self, df: pl.DataFrame) -> dict[str, dict[int, dict[str, float]]]:
        basin_metrics: dict[str, dict[int, dict[str, float]]] = {}
        if df.is_empty():
            return basin_metrics

        unique_basins = df.get_column("basin_id").unique().to_list()
        max_available_horizon = df.get_column("horizon").max()

        for basin_id_val in unique_basins:
            basin_metrics[basin_id_val] = {}
            basin_data = df.filter(pl.col("basin_id") == basin_id_val)
            for h in self.horizons:
                if max_available_horizon is not None and h > max_available_horizon:
                    continue
                horizon_data = basin_data.filter(pl.col("horizon") == h)
                if not horizon_data.is_empty():
                    basin_metrics[basin_id_val][h] = self._calculate_metrics(horizon_data)
                else:
                    basin_metrics[basin_id_val][h] = dict.fromkeys(["MSE", "MAE", "NSE", "RMSE"], np.nan)
        return basin_metrics

    def _calculate_metrics(self, data: pl.DataFrame) -> dict[str, float]:
        if data.is_empty():
            return dict.fromkeys(["MSE", "MAE", "NSE", "RMSE"], np.nan)
        if "prediction" not in data.columns or "observed" not in data.columns:
            return dict.fromkeys(["MSE", "MAE", "NSE", "RMSE"], np.nan)

        # Filter rows where both prediction and observation are non-null to maintain alignment
        valid_data = data.filter(
            pl.col("prediction").is_not_null()
            & pl.col("observed").is_not_null()
            & pl.col("prediction").is_finite()
            & pl.col("observed").is_finite()
        )

        if valid_data.is_empty():
            return dict.fromkeys(["MSE", "MAE", "NSE", "RMSE"], np.nan)

        pred = valid_data.get_column("prediction").to_numpy()
        obs = valid_data.get_column("observed").to_numpy()

        return {
            "MSE": self.calculate_mse(pred, obs),
            "MAE": self.calculate_mae(pred, obs),
            "NSE": self.calculate_nse(pred, obs),
            "RMSE": self.calculate_rmse(pred, obs),
        }

    def summarize_metrics(self, metrics_dict: dict[Any, Any], per_basin: bool = False) -> pl.DataFrame:
        rows = []
        if per_basin:
            for basin_id, basin_data in metrics_dict.items():
                for horizon, horizon_metrics in basin_data.items():
                    rows.append({"basin_id": basin_id, "horizon": horizon, **horizon_metrics})
        else:
            for horizon, horizon_metrics in metrics_dict.items():
                rows.append({"horizon": horizon, **horizon_metrics})

        return pl.from_dicts(rows) if rows else pl.DataFrame()

    def flatten_basin_metrics(self, basin_metrics: dict[str, dict[int, dict[str, float]]]) -> pl.DataFrame:
        rows = []
        for basin_id, horizons_data in basin_metrics.items():
            for horizon, metrics_values in horizons_data.items():
                rows.append({"basin_id": basin_id, "horizon": horizon, **metrics_values})
        return pl.from_dicts(rows) if rows else pl.DataFrame()

    @staticmethod
    def calculate_mse(pred: np.ndarray, obs: np.ndarray) -> float:
        if len(pred) == 0:
            return np.nan
        return np.mean((pred - obs) ** 2)

    @staticmethod
    def calculate_mae(pred: np.ndarray, obs: np.ndarray) -> float:
        if len(pred) == 0:
            return np.nan
        return np.mean(np.abs(pred - obs))

    @staticmethod
    def calculate_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
        if len(pred) == 0:
            return np.nan
        return np.sqrt(np.mean((pred - obs) ** 2))

    @staticmethod
    def calculate_nse(pred: np.ndarray, obs: np.ndarray) -> float:
        if len(obs) == 0 or len(pred) == 0:
            return np.nan
        mean_obs = np.mean(obs)
        if np.all(obs == mean_obs):  # Denominator would be zero
            return 1.0 if np.sum((pred - obs) ** 2) == 0 else -np.inf  # Or np.nan, or other convention
        numerator = np.sum((pred - obs) ** 2)
        denominator = np.sum((obs - mean_obs) ** 2)
        if denominator == 0:  # Should be caught by above, but for safety
            return 1.0 if numerator == 0 else -np.inf
        return 1 - (numerator / denominator)
