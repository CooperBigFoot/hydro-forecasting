import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import pytorch_lightning as pl
import copy


class TSForecastEvaluator:
    """Evaluator for time series forecasting models with per-basin metrics support."""

    def __init__(
        self,
        horizons: List[int],
        models_and_datamodules: Dict[
            str, Tuple[pl.LightningModule, pl.LightningDataModule]
        ] = None,
        default_datamodule=None,  # Optional fallback datamodule
        benchmark_model: str = None,
        trainer_kwargs: Dict = None,
    ):
        """
        Initialize the time series forecast evaluator.

        Args:
            horizons: List of forecast horizons to evaluate (in days)
            models_and_datamodules: Dictionary mapping model names to (model, datamodule) tuples
            default_datamodule: Optional default datamodule to use if no specific one is provided
            benchmark_model: Optional name of the model to use as benchmark for comparisons
            trainer_kwargs: Optional dictionary of kwargs to pass to PyTorch Lightning Trainer
        """
        self.horizons = horizons
        self.default_datamodule = default_datamodule
        self.benchmark_model = benchmark_model
        self.trainer_kwargs = trainer_kwargs or {"accelerator": "cpu", "devices": 1}
        self.results = {}

        # Initialize storage for models and their datamodules
        self.models = {}
        self.datamodules = {}

        # Store models and datamodules if provided
        if models_and_datamodules:
            for name, (model, datamodule) in models_and_datamodules.items():
                self.models[name] = copy.deepcopy(model)
                self.datamodules[name] = datamodule

    def register_model(
        self,
        name: str,
        model: pl.LightningModule,
        datamodule: Optional[pl.LightningDataModule] = None,
    ):
        """
        Register a new model with its specific datamodule.

        Args:
            name: Name identifier for the model
            model: PyTorch Lightning model to register
            datamodule: Optional datamodule specific to this model
        """
        self.models[name] = copy.deepcopy(model)
        if datamodule:
            self.datamodules[name] = datamodule
        elif name not in self.datamodules and self.default_datamodule is None:
            print(
                f"Warning: No datamodule provided for model '{name}' and no default datamodule available"
            )

    def test_models(self, datamodule=None):
        """
        Test all registered models and evaluate results.

        Args:
            datamodule: Optional datamodule to use as fallback if no model-specific datamodule exists

        Returns:
            Dictionary with test results for all models
        """
        for name, model in self.models.items():
            # Determine which datamodule to use in priority order
            model_datamodule = self.datamodules.get(
                name, datamodule or self.default_datamodule
            )

            if model_datamodule is None:
                raise ValueError(f"No datamodule found for model '{name}'")

            print(f"Testing {name}...")

            # Create a trainer and run test
            trainer = pl.Trainer(**self.trainer_kwargs)
            trainer.test(model, datamodule=model_datamodule)

            # Verify model has test results
            if not hasattr(model, "test_results"):
                raise AttributeError(
                    f"Model {name} doesn't have test_results attribute. "
                    "Ensure your LightningModule stores test outputs in self.test_results."
                )

            # Extract and evaluate results
            df, metrics, basin_metrics = self.evaluate(
                model.test_results, model_datamodule
            )
            self.results[name] = {
                "df": df,
                "metrics": metrics,
                "basin_metrics": basin_metrics,
                "datamodule": model_datamodule,  # Store reference to used datamodule
            }

        return self.results

    def test_specific_model(self, name: str, datamodule=None):
        """
        Test a specific model by name.

        Args:
            name: Name of the model to test
            datamodule: Optional datamodule to use for testing

        Returns:
            Dictionary with test results for the specified model
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")

        model = self.models[name]
        model_datamodule = datamodule or self.datamodules.get(
            name, self.default_datamodule
        )

        if model_datamodule is None:
            raise ValueError(f"No datamodule found for model '{name}'")

        trainer = pl.Trainer(**self.trainer_kwargs)
        trainer.test(model, datamodule=model_datamodule)

        df, metrics, basin_metrics = self.evaluate(model.test_results, model_datamodule)

        self.results[name] = {
            "df": df,
            "metrics": metrics,
            "basin_metrics": basin_metrics,
            "datamodule": model_datamodule,
        }

        return self.results[name]

    def evaluate(
        self, test_results: Dict[str, torch.Tensor], datamodule=None
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Evaluate model test results and compute metrics.

        Args:
            test_results: Dictionary containing model outputs and observations
            datamodule: Specific datamodule to use for evaluation

        Returns:
            Tuple containing:
                - DataFrame with predictions, observations, and metadata
                - Dictionary with overall metrics by horizon
                - Dictionary with per-basin metrics by horizon
        """
        # Create evaluation dataframe from test results
        df = self._prepare_evaluation_dataframe(test_results, datamodule)

        # Calculate overall metrics for each horizon
        overall_metrics = self._calculate_overall_metrics(df)

        # Calculate per-basin metrics
        basin_metrics = self._calculate_basin_metrics(df)

        return df, overall_metrics, basin_metrics

    def _prepare_evaluation_dataframe(
        self, test_results: Dict[str, torch.Tensor], datamodule=None
    ) -> pd.DataFrame:
        """
        Create a flattened dataframe with predictions, observations, and metadata.

        Args:
            test_results: Dictionary containing model outputs and observations
            datamodule: Specific datamodule to use for inverse transformations

        Returns:
            DataFrame with predictions, observations, basin IDs, horizons and dates
        """
        # Data extraction
        basin_ids = np.array(test_results["basin_ids"])
        preds = test_results["predictions"].cpu().numpy()
        obs = test_results["observations"].cpu().numpy()

        print(
            f"Evaluating results with shape: preds={preds.shape}, obs={obs.shape}, basin_ids={basin_ids.shape}"
        )

        # Ensure pred and obs dimensions match
        if preds.shape != obs.shape:
            raise ValueError(
                f"Prediction shape {preds.shape} doesn't match observation shape {obs.shape}"
            )

        # Create expanded basin IDs and horizons
        if preds.ndim == 2:  # [batch_size, pred_len]
            horizons_per_sample = preds.shape[1]

            # Handle horizon mismatch - don't modify self.horizons, use a local variable
            current_horizons = self.horizons

            if horizons_per_sample != len(current_horizons):
                print(
                    f"Warning: Model output has {horizons_per_sample} horizons but evaluator configured with {len(current_horizons)} horizons"
                )
                # Use the actual horizons from model output
                current_horizons = list(range(1, horizons_per_sample + 1))
                print(f"Using adjusted horizons: {current_horizons}")

            # Flatten predictions and observations
            preds_flat = preds.flatten()
            obs_flat = obs.flatten()

            # Repeat each basin ID for each horizon in the output
            basin_ids_expanded = np.repeat(basin_ids, horizons_per_sample)

            # Create repeated horizons array matching the model's output structure
            horizons_expanded = np.tile(current_horizons, len(basin_ids))

            # Verify all arrays have matching lengths
            assert (
                len(preds_flat)
                == len(obs_flat)
                == len(basin_ids_expanded)
                == len(horizons_expanded)
            ), (
                f"Array length mismatch: preds_flat={len(preds_flat)}, obs_flat={len(obs_flat)}, "
                f"basin_ids_expanded={len(basin_ids_expanded)}, horizons_expanded={len(horizons_expanded)}"
            )

            # Create dates if available
            if "input_end_date" in test_results:
                input_end_dates = test_results["input_end_date"]

                # Ensure input_end_dates matches basin_ids length
                if len(input_end_dates) != len(basin_ids):
                    print(
                        f"Warning: input_end_dates length ({len(input_end_dates)}) doesn't match basin_ids length ({len(basin_ids)})"
                    )
                    # Adjust to match basin_ids
                    if len(input_end_dates) < len(basin_ids):
                        input_end_dates = input_end_dates + [input_end_dates[-1]] * (
                            len(basin_ids) - len(input_end_dates)
                        )
                    else:
                        input_end_dates = input_end_dates[: len(basin_ids)]

                # Create expanded dates for each horizon - use current_horizons not self.horizons
                dates_expanded = []
                for i, input_date in enumerate(input_end_dates):
                    input_date_dt = pd.to_datetime(input_date)
                    for horizon in current_horizons:
                        # Calculate forecast date by adding horizon days to input end date
                        forecast_date = input_date_dt + pd.Timedelta(days=horizon)
                        dates_expanded.append(forecast_date)

                # Verify dates_expanded length matches other arrays
                assert len(dates_expanded) == len(preds_flat), (
                    f"dates_expanded length {len(dates_expanded)} doesn't match preds_flat length {len(preds_flat)}"
                )
            else:
                # Create dummy dates if not available
                print("Warning: No input_end_dates found, using dummy dates")
                dates_expanded = [pd.Timestamp.now()] * len(preds_flat)
        else:
            raise ValueError(
                f"Unexpected prediction shape {preds.shape}, expected 2D array [batch_size, pred_len]"
            )

        # Use the model-specific datamodule for inverse transforms if provided
        dm_for_transform = datamodule or self.default_datamodule
        if dm_for_transform and hasattr(
            dm_for_transform, "inverse_transform_predictions"
        ):
            try:
                preds_flat = dm_for_transform.inverse_transform_predictions(
                    preds_flat, basin_ids_expanded
                )
                obs_flat = dm_for_transform.inverse_transform_predictions(
                    obs_flat, basin_ids_expanded
                )
            except Exception as e:
                print(f"Warning: Failed to inverse transform predictions: {e}")

        # Create evaluation dataframe
        df = pd.DataFrame(
            {
                "horizon": horizons_expanded,
                "prediction": preds_flat,
                "observed": obs_flat,
                "basin_id": basin_ids_expanded,
                "date": dates_expanded,
            }
        )

        return df

    def _calculate_overall_metrics(
        self, df: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate overall metrics for each forecast horizon.

        Args:
            df: DataFrame with predictions and observations

        Returns:
            Dictionary with metrics for each horizon
        """
        overall_metrics = {}
        for h in self.horizons:
            if h > max(df["horizon"]):
                print(
                    f"Warning: Horizon {h} exceeds maximum available horizon {max(df['horizon'])}"
                )
                continue

            horizon_data = df[df["horizon"] == h]
            if not horizon_data.empty:
                overall_metrics[h] = self._calculate_metrics(horizon_data)
            else:
                print(f"Warning: No data available for horizon {h}")
                overall_metrics[h] = {
                    metric: np.nan for metric in ["MSE", "MAE", "NSE", "RMSE"]
                }

        return overall_metrics

    def _calculate_basin_metrics(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Calculate per-basin metrics for each forecast horizon.

        Args:
            df: DataFrame with predictions and observations

        Returns:
            Nested dictionary with metrics by basin and horizon
        """
        basin_metrics = {}
        for basin in df["basin_id"].unique():
            basin_metrics[basin] = {}
            basin_data = df[df["basin_id"] == basin]

            for h in self.horizons:
                if h > max(df["horizon"]):
                    continue

                horizon_data = basin_data[basin_data["horizon"] == h]
                if not horizon_data.empty:
                    basin_metrics[basin][h] = self._calculate_metrics(horizon_data)
                else:
                    basin_metrics[basin][h] = {
                        metric: np.nan for metric in ["MSE", "MAE", "NSE", "RMSE"]
                    }

        return basin_metrics

    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Helper method to calculate metrics for a subset of data."""
        if len(data) == 0:
            return {metric: np.nan for metric in ["MSE", "MAE", "NSE", "RMSE"]}

        pred = data["prediction"].values
        obs = data["observed"].values

        return {
            "MSE": self.calculate_mse(pred, obs),
            "MAE": self.calculate_mae(pred, obs),
            "NSE": self.calculate_nse(pred, obs),
            "RMSE": self.calculate_rmse(pred, obs),
        }

    def summarize_metrics(self, metrics: Dict, per_basin: bool = False) -> pd.DataFrame:
        """Create a summary DataFrame of metrics."""
        rows = []

        if per_basin:
            for basin, basin_data in metrics.items():
                for horizon, horizon_metrics in basin_data.items():
                    rows.append(
                        {"basin_id": basin, "horizon": horizon, **horizon_metrics}
                    )
            return pd.DataFrame(rows).set_index(["basin_id", "horizon"])

        else:
            for horizon, horizon_metrics in metrics.items():
                rows.append({"horizon": horizon, **horizon_metrics})
            return pd.DataFrame(rows).set_index("horizon")

    def flatten_basin_metrics(self, basin_metrics: Dict) -> pd.DataFrame:
        """
        Convert nested basin metrics dictionary to a flattened DataFrame.

        Args:
            basin_metrics: Nested dictionary of metrics by basin and horizon

        Returns:
            DataFrame with columns for basin_id, horizon, and metrics
        """
        rows = []
        for basin, horizons in basin_metrics.items():
            for horizon, metrics in horizons.items():
                row = {"basin_id": basin, "horizon": horizon}
                row.update(metrics)
                rows.append(row)
        return pd.DataFrame(rows)

    # Static metric calculation methods
    @staticmethod
    def calculate_mse(pred: np.ndarray, obs: np.ndarray) -> float:
        return np.mean((pred - obs) ** 2)

    @staticmethod
    def calculate_mae(pred: np.ndarray, obs: np.ndarray) -> float:
        return np.mean(np.abs(pred - obs))

    @staticmethod
    def calculate_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
        return np.sqrt(np.mean((pred - obs) ** 2))

    @staticmethod
    def calculate_nse(pred: np.ndarray, obs: np.ndarray) -> float:
        return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
