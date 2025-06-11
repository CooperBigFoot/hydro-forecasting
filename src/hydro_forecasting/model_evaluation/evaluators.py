import logging
import pickle
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
    """Evaluator for time series forecasting models with horizon-specific metrics.

    This class tests pre-trained PyTorch Lightning models and computes performance
    metrics separately for each forecast horizon (e.g., 1-day ahead, 2-day ahead, etc.).
    """

    def __init__(
        self,
        horizons: list[int],
        models_and_datamodules: dict[str, tuple],
        trainer_kwargs: dict[str, Any],
        save_path: str | Path | None = None,
    ) -> None:
        """Initialize the time series forecast evaluator.

        Args:
            horizons: List of forecast horizons to evaluate (e.g., [1, 2, 3, ..., 10])
            models_and_datamodules: Dictionary mapping model names to (model, datamodule) tuples
            trainer_kwargs: PyTorch Lightning Trainer configuration dictionary
            save_path: Optional path to save evaluation results
        """
        self.horizons = horizons
        self.models_and_datamodules = models_and_datamodules
        self.trainer_kwargs = trainer_kwargs
        self.save_path = Path(save_path) if save_path else None

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

    def test_models(self) -> dict[str, dict]:
        """Test all models and compute horizon-specific metrics.

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
        results = {}

        for model_name, (model, datamodule) in self.models_and_datamodules.items():
            self.logger.info(f"Testing model: {model_name}")

            try:
                model_results = self._test_single_model(model, datamodule, model_name)
                results[model_name] = model_results
                self.logger.info(f"Successfully tested model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to test model {model_name}: {e}")
                continue

        # Save results if path provided
        if self.save_path:
            self._save_results(results)

        return results

    def _test_single_model(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule, model_name: str
    ) -> dict[str, Any]:
        """Test a single model and process results.

        Args:
            model: PyTorch Lightning model to test
            datamodule: Associated data module
            model_name: Name of the model for logging

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
        metrics_by_gauge = self._calculate_metrics_by_gauge(df)

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
                if input_end_date is not None:
                    forecast_date = input_end_date + pd.Timedelta(days=horizon)
                else:
                    forecast_date = None

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

    def _calculate_metrics_by_gauge(self, df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate metrics for each gauge and horizon combination.

        Args:
            df: DataFrame with predictions and observations

        Returns:
            Nested dictionary with structure: {gauge_id: {horizon_X: {metric: value}}}
        """
        metrics_by_gauge = {}

        for gauge_id in df["gauge_id"].unique():
            gauge_df = df[df["gauge_id"] == gauge_id]
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

                metrics_by_gauge[gauge_id][f"horizon_{horizon}"] = horizon_metrics

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
