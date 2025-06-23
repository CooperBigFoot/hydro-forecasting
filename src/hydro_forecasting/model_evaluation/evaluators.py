import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from .metrics import (
    calculate_atpe,
    calculate_kge,
    calculate_mae,
    calculate_mse,
    calculate_nse,
    calculate_pbias,
    calculate_pearson_r,
    calculate_rmse,
)


class TSForecastEvaluator:
    """Evaluator for time series forecasting models with horizon-specific metrics and caching.

    This class tests pre-trained PyTorch Lightning models and computes performance
    metrics separately for each forecast horizon (e.g., 1-day ahead, 2-day ahead, etc.).
    Results can be cached to disk for faster subsequent evaluations.
    """

    def __init__(
        self,
        horizons: list[int],
        models_and_datamodules: dict[str, tuple],
        trainer_kwargs: dict[str, Any],
        save_path: str | Path | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        """Initialize the time series forecast evaluator.

        Args:
            horizons: List of forecast horizons to evaluate (e.g., [1, 2, 3, ..., 10])
            models_and_datamodules: Dictionary mapping model names to (model, datamodule) tuples
            trainer_kwargs: PyTorch Lightning Trainer configuration dictionary
            save_path: Optional path to save evaluation results (legacy pickle format)
            cache_path: Optional path to cache directory for faster subsequent evaluations
        """
        self.horizons = horizons
        self.models_and_datamodules = models_and_datamodules
        self.trainer_kwargs = trainer_kwargs
        self.save_path = Path(save_path) if save_path else None
        self.cache_path = Path(cache_path) if cache_path else None

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Define all metrics functions
        self.metric_functions = {
            "mse": calculate_mse,
            "mae": calculate_mae,
            "rmse": calculate_rmse,
            "nse": calculate_nse,
            "pearson_r": calculate_pearson_r,
            "kge": calculate_kge,
            "pbias": calculate_pbias,
            "atpe": calculate_atpe,
        }

    def test_models(
        self,
        start_of_season: int | None = None,
        end_of_season: int | None = None,
        cache_path: str | Path | None = None,
        force_refresh: bool = False,
    ) -> dict[str, dict]:
        """Test all models and compute horizon-specific metrics with caching support.

        Args:
            start_of_season: Starting month (1-12) for seasonal evaluation. If None, use all data.
            end_of_season: Ending month (1-12) for seasonal evaluation (exclusive). If None, use all data.
            cache_path: Cache directory path (overrides constructor cache_path if provided)
            force_refresh: If True, ignore cache and re-run all model tests

        Returns:
            Dictionary containing evaluation results for each model with structure:
            {
                "model_name": {
                    "predictions_df": pd.DataFrame,  # columns: horizon, observed, predicted, date, gauge_id
                    "metrics_by_gauge": {
                        "gauge_id": {
                            "horizon_1": {"mse": float, "nse": float, ...},
                            "horizon_2": {"mse": float, "nse": float, ...},
                            ...
                        }
                    }
                }
            }
        """
        # Determine which cache path to use
        active_cache_path = Path(cache_path) if cache_path else self.cache_path

        results = {}

        # If caching is enabled and not forcing refresh, check cache first
        if active_cache_path and not force_refresh:
            # Create cache directory if it doesn't exist
            self._create_cache_directory(active_cache_path)

            # Load cache metadata
            cached_models = self._load_cache_metadata(active_cache_path)

            self.logger.info(f"Cache found at {active_cache_path}")
            if cached_models:
                self.logger.info(f"Found {len(cached_models)} cached models")

        else:
            cached_models = {}

        # Determine which models need testing
        models_to_test = []
        models_to_load = []

        for model_name in self.models_and_datamodules:
            cache_key = self._generate_cache_key(model_name, start_of_season, end_of_season)

            if not force_refresh and active_cache_path and cache_key in cached_models:
                models_to_load.append(model_name)
            else:
                models_to_test.append(model_name)

        # Load cached results
        if models_to_load:
            self.logger.info(
                f"Loading cached results for: {', '.join(models_to_load)} ({len(models_to_load)}/{len(self.models_and_datamodules)} models)"
            )

            for model_name in models_to_load:
                try:
                    cached_result = self._load_cached_model_results(active_cache_path, model_name)
                    results[model_name] = cached_result
                    self.logger.debug(f"Successfully loaded cached results for {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load cached results for {model_name}: {e}. Will re-run test.")
                    models_to_test.append(model_name)

        # Test models that aren't cached or failed to load
        if models_to_test:
            self.logger.info(
                f"Testing models: {', '.join(models_to_test)} ({len(models_to_test)}/{len(self.models_and_datamodules)} models)"
            )

            for model_name in models_to_test:
                model, datamodule = self.models_and_datamodules[model_name]
                self.logger.info(f"Testing model: {model_name}")

                try:
                    model_results = self._test_single_model(
                        model, datamodule, model_name, start_of_season, end_of_season
                    )
                    results[model_name] = model_results

                    # Save to cache if caching is enabled
                    if active_cache_path:
                        self._save_model_to_cache(
                            active_cache_path,
                            model_name,
                            model_results["predictions_df"],
                            model_results["metrics_by_gauge"],
                            start_of_season,
                            end_of_season,
                        )
                        self.logger.info(f"Cached {model_name} results to {active_cache_path}")

                    self.logger.info(f"Successfully tested model: {model_name}")

                except Exception as e:
                    self.logger.error(f"Failed to test model {model_name}: {e}")
                    continue

        # Save results to legacy pickle format if path provided
        if self.save_path:
            self._save_results(results)

        return results

    def _create_cache_directory(self, cache_path: Path) -> None:
        """Create cache directory structure if it doesn't exist.

        Args:
            cache_path: Path to cache directory
        """
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "predictions").mkdir(exist_ok=True)
            (cache_path / "metrics").mkdir(exist_ok=True)
            self.logger.debug(f"Created cache directory structure at {cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to create cache directory {cache_path}: {e}")
            raise

    def _generate_cache_key(self, model_name: str, start_of_season: int | None, end_of_season: int | None) -> str:
        """Generate a unique cache key for a model with specific parameters.

        Args:
            model_name: Name of the model
            start_of_season: Starting month for seasonal evaluation
            end_of_season: Ending month for seasonal evaluation

        Returns:
            Unique cache key string
        """
        horizons_str = ",".join(map(str, sorted(self.horizons)))
        season_str = f"{start_of_season}-{end_of_season}" if start_of_season is not None else "all"
        return f"{model_name}_h[{horizons_str}]_s[{season_str}]"

    def _load_cache_metadata(self, cache_path: Path) -> dict[str, dict]:
        """Load cache metadata from disk.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary of cached model information
        """
        metadata_file = cache_path / "cache_metadata.json"

        if not metadata_file.exists():
            self.logger.debug("No cache metadata found, starting with empty cache")
            return {}

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                return metadata.get("cached_models", {})
        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}. Starting with empty cache.")
            return {}

    def _load_cached_model_results(self, cache_path: Path, model_name: str) -> dict[str, Any]:
        """Load cached results for a specific model.

        Args:
            cache_path: Path to cache directory
            model_name: Name of the model to load

        Returns:
            Dictionary containing predictions_df and metrics_by_gauge

        Raises:
            Exception: If cached files cannot be loaded
        """
        # Load predictions
        predictions_file = cache_path / "predictions" / f"{model_name}.parquet"
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        predictions_df = pd.read_parquet(predictions_file)

        # Load metrics
        metrics_file = cache_path / "metrics" / f"{model_name}.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        with open(metrics_file) as f:
            metrics_by_gauge = json.load(f)

        # Convert string keys back to integers for horizons
        processed_metrics = {}
        for gauge_id, gauge_data in metrics_by_gauge.items():
            processed_metrics[gauge_id] = {}
            for horizon_str, metrics in gauge_data.items():
                horizon_int = int(horizon_str)
                processed_metrics[gauge_id][horizon_int] = metrics

        return {"predictions_df": predictions_df, "metrics_by_gauge": processed_metrics}

    def _save_model_to_cache(
        self,
        cache_path: Path,
        model_name: str,
        predictions_df: pd.DataFrame,
        metrics_by_gauge: dict,
        start_of_season: int | None,
        end_of_season: int | None,
    ) -> None:
        """Save model results to cache.

        Args:
            cache_path: Path to cache directory
            model_name: Name of the model
            predictions_df: Predictions DataFrame
            metrics_by_gauge: Metrics dictionary
            start_of_season: Starting month for seasonal evaluation
            end_of_season: Ending month for seasonal evaluation
        """
        try:
            # Save predictions to parquet
            predictions_file = cache_path / "predictions" / f"{model_name}.parquet"
            predictions_df.to_parquet(predictions_file, compression="snappy", index=False)

            # Convert numpy types to Python types for JSON serialization
            json_safe_metrics = self._convert_to_json_safe(metrics_by_gauge)

            # Save metrics to JSON
            metrics_file = cache_path / "metrics" / f"{model_name}.json"
            with open(metrics_file, "w") as f:
                json.dump(json_safe_metrics, f, indent=2)

            # Update cache metadata
            self._update_cache_metadata(cache_path, model_name, predictions_df, start_of_season, end_of_season)

            self.logger.debug(f"Saved {model_name} to cache: {predictions_file.stat().st_size / 1024 / 1024:.1f} MB")

        except Exception as e:
            self.logger.error(f"Failed to save {model_name} to cache: {e}")
            raise

    def _convert_to_json_safe(self, obj: Any) -> Any:
        """Convert numpy types to JSON-safe Python types recursively.

        Args:
            obj: Object to convert

        Returns:
            JSON-safe version of the object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _update_cache_metadata(
        self,
        cache_path: Path,
        model_name: str,
        predictions_df: pd.DataFrame,
        start_of_season: int | None,
        end_of_season: int | None,
    ) -> None:
        """Update cache metadata with information about a newly cached model.

        Args:
            cache_path: Path to cache directory
            model_name: Name of the model
            predictions_df: Predictions DataFrame for statistics
            start_of_season: Starting month for seasonal evaluation
            end_of_season: Ending month for seasonal evaluation
        """
        metadata_file = cache_path / "cache_metadata.json"

        # Load existing metadata or create new
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}

        # Initialize structure if needed
        if "cache_version" not in metadata:
            metadata["cache_version"] = "1.0"
            metadata["created_at"] = datetime.now().isoformat()

        if "cached_models" not in metadata:
            metadata["cached_models"] = {}

        # Generate cache key
        cache_key = self._generate_cache_key(model_name, start_of_season, end_of_season)

        # Calculate file size
        predictions_file = cache_path / "predictions" / f"{model_name}.parquet"
        file_size_mb = predictions_file.stat().st_size / 1024 / 1024 if predictions_file.exists() else 0

        # Update metadata for this model
        metadata["cached_models"][cache_key] = {
            "model_name": model_name,
            "cached_at": datetime.now().isoformat(),
            "horizons": sorted(self.horizons),
            "start_of_season": start_of_season,
            "end_of_season": end_of_season,
            "num_basins": len(predictions_df["gauge_id"].unique()) if not predictions_df.empty else 0,
            "num_predictions": len(predictions_df),
            "file_size_mb": round(file_size_mb, 2),
        }

        metadata["last_updated"] = datetime.now().isoformat()

        # Save updated metadata
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update cache metadata: {e}")

    def _test_single_model(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        model_name: str,
        start_of_season: int | None = None,
        end_of_season: int | None = None,
    ) -> dict[str, Any]:
        """Test a single model and process results.

        Args:
            model: PyTorch Lightning model to test
            datamodule: Associated data module
            model_name: Name of the model for logging
            start_of_season: Starting month (1-12) for seasonal evaluation
            end_of_season: Ending month (1-12) for seasonal evaluation (exclusive)

        Returns:
            Dictionary containing predictions DataFrame and metrics

        Raises:
            RuntimeError: If no test results are found
            ValueError: If prediction and observation shapes don't match
        """
        # Create trainer and run test
        trainer = pl.Trainer(**self.trainer_kwargs)
        trainer.test(model, datamodule)

        # Extract test results
        test_results = model.test_results
        if not test_results:
            raise RuntimeError(f"No test results found for model {model_name}")

        # Extract components
        predictions = test_results["predictions"]  # [N_samples, output_len]
        observations = test_results["observations"]  # [N_samples, output_len]
        basin_ids = test_results["basin_ids"]  # [N_samples]
        input_end_dates = test_results.get("input_end_date", [None] * len(basin_ids))

        # Validate shapes
        if observations.shape != predictions.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}")

        # Convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(observations):
            observations = observations.detach().cpu().numpy()

        # Apply inverse transformation to both predictions and observations
        predictions_inv = self._apply_inverse_transform(predictions, basin_ids, datamodule)
        observations_inv = self._apply_inverse_transform(observations, basin_ids, datamodule)

        # Reshape data to horizon-wise format
        df = self._reshape_to_horizon_format(predictions_inv, observations_inv, basin_ids, input_end_dates)

        # Calculate metrics by gauge and horizon
        metrics_by_gauge = self._calculate_metrics_by_gauge(df, start_of_season, end_of_season)

        return {"predictions_df": df, "metrics_by_gauge": metrics_by_gauge}

    def _apply_inverse_transform(
        self, data: np.ndarray, basin_ids: list[str], datamodule: pl.LightningDataModule
    ) -> np.ndarray:
        """Apply inverse transformation to predictions or observations.

        Args:
            data: Data array with shape [N_samples, output_len]
            basin_ids: List of basin identifiers
            datamodule: Data module with inverse transformation method

        Returns:
            Inverse transformed data with same shape as input
        """
        try:
            return datamodule.inverse_transform_predictions(data, np.array(basin_ids))
        except Exception as e:
            self.logger.warning(f"Inverse transformation failed: {e}. Using original data.")
            return data

    def _reshape_to_horizon_format(
        self, predictions: np.ndarray, observations: np.ndarray, basin_ids: list[str], input_end_dates: list[int | None]
    ) -> pd.DataFrame:
        """Reshape data from [N_samples, output_len] to horizon-wise DataFrame.

        Args:
            predictions: Predictions array [N_samples, output_len]
            observations: Observations array [N_samples, output_len]
            basin_ids: List of basin identifiers [N_samples]
            input_end_dates: List of input end dates in milliseconds [N_samples]

        Returns:
            DataFrame with columns ["horizon", "observed", "predicted", "date", "gauge_id"]
        """
        rows = []
        n_samples, output_len = predictions.shape

        for sample_idx in range(n_samples):
            basin_id = basin_ids[sample_idx]
            input_end_date_ms = input_end_dates[sample_idx]

            # Convert input end date to timestamp if available
            input_end_date = None
            if input_end_date_ms is not None:
                # Convert tensor to scalar if needed
                if torch.is_tensor(input_end_date_ms):
                    input_end_date_ms = input_end_date_ms.item()
                input_end_date = pd.Timestamp(input_end_date_ms, unit="ms")

            for horizon_idx in range(output_len):
                horizon = horizon_idx + 1  # 1-indexed horizons

                # Only include horizons we're interested in
                if horizon not in self.horizons:
                    continue

                pred_val = predictions[sample_idx, horizon_idx]
                obs_val = observations[sample_idx, horizon_idx]

                # Calculate forecast date
                forecast_date = input_end_date + pd.Timedelta(days=horizon) if input_end_date is not None else None

                # Skip rows with missing data
                if pd.isna(pred_val) or pd.isna(obs_val):
                    continue

                rows.append(
                    {
                        "horizon": horizon,
                        "observed": obs_val,
                        "predicted": pred_val,
                        "date": forecast_date,
                        "gauge_id": basin_id,
                    }
                )

        return pd.DataFrame(rows)

    def _calculate_metrics_by_gauge(
        self, df: pd.DataFrame, start_of_season: int | None = None, end_of_season: int | None = None
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate metrics for each gauge and horizon combination.

        Args:
            df: DataFrame with predictions and observations
            start_of_season: Starting month (1-12) for seasonal evaluation
            end_of_season: Ending month (1-12) for seasonal evaluation (exclusive)

        Returns:
            Nested dictionary with structure: {gauge_id: {horizon_X: {metric: value}}}
        """
        # Apply seasonal filtering if specified
        if start_of_season is not None and end_of_season is not None:
            # Create filtered DataFrame for metric calculations
            seasonal_mask = (df["date"].dt.month >= start_of_season) & (df["date"].dt.month < end_of_season)
            df_for_metrics = df[seasonal_mask].copy()
            self.logger.info(f"Applying seasonal filter: months {start_of_season} to {end_of_season - 1}")
        else:
            # Use original DataFrame for metric calculations
            df_for_metrics = df

        metrics_by_gauge = {}

        for gauge_id in df_for_metrics["gauge_id"].unique():
            gauge_df = df_for_metrics[df_for_metrics["gauge_id"] == gauge_id]
            metrics_by_gauge[gauge_id] = {}

            for horizon in self.horizons:
                horizon_df = gauge_df[gauge_df["horizon"] == horizon]

                if horizon_df.empty:
                    continue

                observed = horizon_df["observed"].values
                predicted = horizon_df["predicted"].values

                # Calculate all metrics
                horizon_metrics = {}
                for metric_name, metric_func in self.metric_functions.items():
                    try:
                        metric_value = metric_func(predicted, observed)
                        horizon_metrics[metric_name] = metric_value
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate {metric_name} for {gauge_id} horizon {horizon}: {e}")
                        horizon_metrics[metric_name] = np.nan

                metrics_by_gauge[gauge_id][horizon] = horizon_metrics

        return metrics_by_gauge

    def _save_results(self, results: dict[str, dict]) -> None:
        """Save evaluation results to disk using pickle.

        Args:
            results: Evaluation results dictionary
        """
        try:
            if self.save_path is not None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.save_path, "wb") as f:
                    pickle.dump(results, f)

                self.logger.info(f"Results saved to {self.save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
