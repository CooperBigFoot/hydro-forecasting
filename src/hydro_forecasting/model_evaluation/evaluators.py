import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    Results can be cached to disk for faster subsequent evaluations with comprehensive
    error handling and validation.
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
        # Validate input parameters
        self._validate_input_parameters(start_of_season, end_of_season)

        # Determine which cache path to use
        active_cache_path = Path(cache_path) if cache_path else self.cache_path

        results = {}

        # If caching is enabled and not forcing refresh, check cache first
        if active_cache_path and not force_refresh:
            # Attempt to initialize cache with comprehensive error handling
            cache_available = self._initialize_cache_safely(active_cache_path)

            if cache_available:
                # Load cache metadata with validation
                cached_models = self._load_cache_metadata_safely(active_cache_path)

                if cached_models:
                    self.logger.info(f"Cache found at {active_cache_path} with {len(cached_models)} cached models")
                else:
                    self.logger.info(f"Cache initialized at {active_cache_path} (empty)")
            else:
                self.logger.warning(f"Cache unavailable at {active_cache_path}, proceeding without caching")
                cached_models = {}
                active_cache_path = None
        else:
            cached_models = {}
            if force_refresh:
                self.logger.info("Force refresh enabled, ignoring all cached results")

        # Determine which models need testing vs loading from cache
        models_to_test = []
        models_to_load = []

        for model_name in self.models_and_datamodules.keys():
            if not force_refresh and active_cache_path:
                cache_key = self._generate_cache_key(model_name, start_of_season, end_of_season)

                if cache_key in cached_models:
                    # Validate cache parameters match current request
                    if self._validate_cache_parameters(cached_models[cache_key], start_of_season, end_of_season):
                        models_to_load.append(model_name)
                        self.logger.debug(
                            f"Cache hit for {model_name} (horizons={self.horizons}, season={start_of_season}-{end_of_season})"
                        )
                    else:
                        models_to_test.append(model_name)
                        self.logger.info(f"Cache parameters mismatch for {model_name}, re-running test")
                else:
                    models_to_test.append(model_name)
                    self.logger.debug(f"Cache miss for {model_name}")
            else:
                models_to_test.append(model_name)

        # Load cached results with comprehensive error handling
        if models_to_load:
            self.logger.info(
                f"Loading cached results for: {', '.join(models_to_load)} ({len(models_to_load)}/{len(self.models_and_datamodules)} models)"
            )

            for model_name in models_to_load:
                try:
                    cached_result = self._load_cached_model_results_safely(active_cache_path, model_name)

                    # Validate loaded data integrity
                    if self._validate_cached_data(cached_result, model_name):
                        results[model_name] = cached_result
                        self.logger.debug(f"Successfully loaded and validated cached results for {model_name}")
                    else:
                        self.logger.warning(f"Cached data validation failed for {model_name}, re-running test")
                        models_to_test.append(model_name)

                except Exception as e:
                    self.logger.warning(f"Failed to load cached results for {model_name}: {e}. Re-running test.")
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

                    # Save to cache if caching is available
                    if active_cache_path:
                        try:
                            self._save_model_to_cache_safely(
                                active_cache_path,
                                model_name,
                                model_results["predictions_df"],
                                model_results["metrics_by_gauge"],
                                start_of_season,
                                end_of_season,
                            )

                            # Calculate file size for logging
                            predictions_file = active_cache_path / "predictions" / f"{model_name}.parquet"
                            file_size_mb = (
                                predictions_file.stat().st_size / 1024 / 1024 if predictions_file.exists() else 0
                            )
                            self.logger.info(
                                f"Cached {model_name} results ({file_size_mb:.1f} MB) to {active_cache_path}"
                            )

                        except Exception as cache_error:
                            self.logger.error(f"Failed to cache {model_name} results: {cache_error}")
                            # Continue without caching - don't fail the entire evaluation

                    self.logger.info(f"Successfully tested model: {model_name}")

                except Exception as e:
                    self.logger.error(f"Failed to test model {model_name}: {e}")
                    continue

        # Save results to legacy pickle format if path provided
        if self.save_path:
            self._save_results(results)

        self.logger.info(f"Evaluation completed: {len(results)}/{len(self.models_and_datamodules)} models successful")
        return results

    def _validate_input_parameters(self, start_of_season: int | None, end_of_season: int | None) -> None:
        """Validate input parameters for seasonal evaluation.

        Args:
            start_of_season: Starting month (1-12) for seasonal evaluation
            end_of_season: Ending month (1-12) for seasonal evaluation (exclusive)

        Raises:
            ValueError: If parameters are invalid
        """
        if start_of_season is not None:
            if not isinstance(start_of_season, int) or not (1 <= start_of_season <= 12):
                raise ValueError(f"start_of_season must be an integer between 1-12, got: {start_of_season}")

        if end_of_season is not None:
            if not isinstance(end_of_season, int) or not (1 <= end_of_season <= 12):
                raise ValueError(f"end_of_season must be an integer between 1-12, got: {end_of_season}")

        if (start_of_season is not None) != (end_of_season is not None):
            raise ValueError("Both start_of_season and end_of_season must be provided together or both None")

    def _initialize_cache_safely(self, cache_path: Path) -> bool:
        """Safely initialize cache directory with comprehensive error handling.

        Args:
            cache_path: Path to cache directory

        Returns:
            True if cache is available, False otherwise
        """
        try:
            # Check if we can create/access the cache directory
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "predictions").mkdir(exist_ok=True)
            (cache_path / "metrics").mkdir(exist_ok=True)

            # Test write permissions by creating a temporary file
            test_file = cache_path / ".cache_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                self.logger.debug(f"Cache directory initialized successfully at {cache_path}")
                return True
            except PermissionError:
                self.logger.error(f"No write permission for cache directory: {cache_path}")
                return False

        except PermissionError:
            self.logger.error(f"Cannot create cache directory due to permissions: {cache_path}")
            return False
        except OSError as e:
            self.logger.error(f"File system error creating cache directory {cache_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error initializing cache {cache_path}: {e}")
            return False

    def _load_cache_metadata_safely(self, cache_path: Path) -> dict[str, dict]:
        """Load cache metadata with comprehensive error handling and validation.

        Args:
            cache_path: Path to cache directory

        Returns:
            Dictionary of cached model information, empty dict if loading fails
        """
        metadata_file = cache_path / "cache_metadata.json"

        if not metadata_file.exists():
            self.logger.debug("No cache metadata found, starting with empty cache")
            return {}

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Validate metadata structure
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")

            if "cached_models" not in metadata:
                self.logger.warning("Cache metadata missing 'cached_models' field, treating as empty")
                return {}

            cached_models = metadata["cached_models"]
            if not isinstance(cached_models, dict):
                raise ValueError("'cached_models' must be a dictionary")

            # Validate each cached model entry
            validated_models = {}
            for cache_key, model_info in cached_models.items():
                if self._validate_cache_metadata_entry(cache_key, model_info):
                    validated_models[cache_key] = model_info
                else:
                    self.logger.warning(f"Invalid cache metadata entry for {cache_key}, skipping")

            return validated_models

        except json.JSONDecodeError as e:
            self.logger.error(f"Cache metadata corrupted (invalid JSON): {e}")
            return self._handle_corrupted_metadata(cache_path)
        except (ValueError, KeyError) as e:
            self.logger.error(f"Cache metadata has invalid structure: {e}")
            return self._handle_corrupted_metadata(cache_path)
        except PermissionError:
            self.logger.error(f"No read permission for cache metadata: {metadata_file}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error loading cache metadata: {e}")
            return {}

    def _validate_cache_metadata_entry(self, cache_key: str, model_info: dict) -> bool:
        """Validate a single cache metadata entry.

        Args:
            cache_key: Cache key string
            model_info: Model information dictionary

        Returns:
            True if entry is valid, False otherwise
        """
        required_fields = ["model_name", "cached_at", "horizons"]

        try:
            # Check required fields exist
            for field in required_fields:
                if field not in model_info:
                    self.logger.debug(f"Cache entry {cache_key} missing required field: {field}")
                    return False

            # Validate horizons field
            horizons = model_info["horizons"]
            if not isinstance(horizons, list) or not all(isinstance(h, int) for h in horizons):
                self.logger.debug(f"Cache entry {cache_key} has invalid horizons format")
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error validating cache entry {cache_key}: {e}")
            return False

    def _validate_cache_parameters(
        self, cached_info: dict, start_of_season: int | None, end_of_season: int | None
    ) -> bool:
        """Validate that cached model parameters match current request.

        Args:
            cached_info: Cached model information
            start_of_season: Current request start of season
            end_of_season: Current request end of season

        Returns:
            True if parameters match, False otherwise
        """
        try:
            # Check horizons match
            cached_horizons = set(cached_info.get("horizons", []))
            current_horizons = set(self.horizons)

            if cached_horizons != current_horizons:
                self.logger.debug(
                    f"Horizon mismatch: cached={sorted(cached_horizons)}, current={sorted(current_horizons)}"
                )
                return False

            # Check seasonal parameters match
            cached_start = cached_info.get("start_of_season")
            cached_end = cached_info.get("end_of_season")

            if cached_start != start_of_season or cached_end != end_of_season:
                self.logger.debug(
                    f"Season mismatch: cached=({cached_start}-{cached_end}), current=({start_of_season}-{end_of_season})"
                )
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Error validating cache parameters: {e}")
            return False

    def _handle_corrupted_metadata(self, cache_path: Path) -> dict[str, dict]:
        """Handle corrupted cache metadata by backing up and starting fresh.

        Args:
            cache_path: Path to cache directory

        Returns:
            Empty dictionary (fresh start)
        """
        try:
            metadata_file = cache_path / "cache_metadata.json"

            if metadata_file.exists():
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = cache_path / f"cache_metadata_corrupted_{timestamp}.json.bak"

                metadata_file.rename(backup_file)
                self.logger.info(f"Corrupted cache metadata backed up to: {backup_file}")

            self.logger.info("Starting with fresh cache metadata")
            return {}

        except Exception as e:
            self.logger.error(f"Failed to backup corrupted metadata: {e}")
            return {}

    def _load_cached_model_results_safely(self, cache_path: Path, model_name: str) -> dict[str, Any]:
        """Load cached results for a specific model with comprehensive error handling.

        Args:
            cache_path: Path to cache directory
            model_name: Name of the model to load

        Returns:
            Dictionary containing predictions_df and metrics_by_gauge

        Raises:
            Exception: If cached files cannot be loaded or are corrupted
        """
        predictions_file = cache_path / "predictions" / f"{model_name}.parquet"
        metrics_file = cache_path / "metrics" / f"{model_name}.json"

        # Check if files exist
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        try:
            # Load predictions with validation
            predictions_df = pd.read_parquet(predictions_file)
            self._validate_predictions_dataframe(predictions_df, model_name)

            # Load metrics with validation
            with open(metrics_file) as f:
                metrics_by_gauge = json.load(f)
            self._validate_metrics_data(metrics_by_gauge, model_name)

            # Convert string keys back to integers for horizons
            processed_metrics = {}
            for gauge_id, gauge_data in metrics_by_gauge.items():
                processed_metrics[gauge_id] = {}
                for horizon_str, metrics in gauge_data.items():
                    try:
                        horizon_int = int(horizon_str)
                        processed_metrics[gauge_id][horizon_int] = metrics
                    except ValueError:
                        self.logger.warning(f"Invalid horizon key '{horizon_str}' in metrics for {model_name}")
                        continue

            self.logger.debug(
                f"Loaded cached data for {model_name}: {len(predictions_df)} predictions, {len(processed_metrics)} basins"
            )

            return {"predictions_df": predictions_df, "metrics_by_gauge": processed_metrics}

        except pd.errors.ParquetError as e:
            raise Exception(f"Corrupted predictions file for {model_name}: {e}") from e
        except json.JSONDecodeError as e:
            raise Exception(f"Corrupted metrics file for {model_name}: {e}") from e
        except PermissionError as e:
            raise Exception(f"Permission denied accessing cached files for {model_name}: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error loading cached data for {model_name}: {e}") from e

    def _validate_predictions_dataframe(self, df: pd.DataFrame, model_name: str) -> None:
        """Validate loaded predictions DataFrame structure and content.

        Args:
            df: Predictions DataFrame to validate
            model_name: Model name for error messages

        Raises:
            ValueError: If DataFrame structure is invalid
        """
        required_columns = {"horizon", "observed", "predicted", "date", "gauge_id"}

        if df.empty:
            raise ValueError(f"Empty predictions DataFrame for {model_name}")

        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in predictions for {model_name}: {missing}")

        # Validate data types
        if not pd.api.types.is_integer_dtype(df["horizon"]):
            raise ValueError(f"Invalid horizon column type for {model_name}: expected integer")

        if not pd.api.types.is_numeric_dtype(df["observed"]):
            raise ValueError(f"Invalid observed column type for {model_name}: expected numeric")

        if not pd.api.types.is_numeric_dtype(df["predicted"]):
            raise ValueError(f"Invalid predicted column type for {model_name}: expected numeric")

        # Validate horizons match expected
        df_horizons = set(df["horizon"].unique())
        expected_horizons = set(self.horizons)

        if not df_horizons.issubset(expected_horizons):
            unexpected = df_horizons - expected_horizons
            raise ValueError(f"Unexpected horizons in cached data for {model_name}: {unexpected}")

    def _validate_metrics_data(self, metrics: dict, model_name: str) -> None:
        """Validate loaded metrics data structure.

        Args:
            metrics: Metrics dictionary to validate
            model_name: Model name for error messages

        Raises:
            ValueError: If metrics structure is invalid
        """
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics must be a dictionary for {model_name}")

        if not metrics:
            raise ValueError(f"Empty metrics data for {model_name}")

        # Validate structure of at least one gauge
        for gauge_id, gauge_metrics in metrics.items():
            if not isinstance(gauge_metrics, dict):
                raise ValueError(f"Gauge metrics must be dictionaries for {model_name}")

            # Check at least one horizon exists
            if not gauge_metrics:
                continue

            # Validate at least one horizon's metrics
            for horizon_str, horizon_metrics in gauge_metrics.items():
                if not isinstance(horizon_metrics, dict):
                    raise ValueError(f"Horizon metrics must be dictionaries for {model_name}")

                # Check for expected metric names
                expected_metrics = set(self.metric_functions.keys())
                actual_metrics = set(horizon_metrics.keys())

                if not actual_metrics.intersection(expected_metrics):
                    raise ValueError(f"No valid metrics found for {model_name}")

                break  # Only validate first horizon
            break  # Only validate first gauge

    def _validate_cached_data(self, cached_result: dict, model_name: str) -> bool:
        """Validate overall integrity of cached model results.

        Args:
            cached_result: Cached result dictionary
            model_name: Model name for logging

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check structure
            if not isinstance(cached_result, dict):
                self.logger.warning(f"Cached result is not a dictionary for {model_name}")
                return False

            required_keys = {"predictions_df", "metrics_by_gauge"}
            if not required_keys.issubset(cached_result.keys()):
                missing = required_keys - set(cached_result.keys())
                self.logger.warning(f"Missing required keys in cached result for {model_name}: {missing}")
                return False

            predictions_df = cached_result["predictions_df"]
            metrics_by_gauge = cached_result["metrics_by_gauge"]

            # Basic consistency checks
            if predictions_df.empty and metrics_by_gauge:
                self.logger.warning(
                    f"Inconsistent cached data for {model_name}: empty predictions but non-empty metrics"
                )
                return False

            if not predictions_df.empty and not metrics_by_gauge:
                self.logger.warning(
                    f"Inconsistent cached data for {model_name}: non-empty predictions but empty metrics"
                )
                return False

            # Check gauge consistency
            pred_gauges = set(predictions_df["gauge_id"].unique()) if not predictions_df.empty else set()
            metric_gauges = set(metrics_by_gauge.keys())

            if pred_gauges != metric_gauges:
                self.logger.warning(
                    f"Gauge mismatch in cached data for {model_name}: pred={len(pred_gauges)}, metrics={len(metric_gauges)}"
                )
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Error validating cached data for {model_name}: {e}")
            return False

    def _save_model_to_cache_safely(
        self,
        cache_path: Path,
        model_name: str,
        predictions_df: pd.DataFrame,
        metrics_by_gauge: dict,
        start_of_season: int | None,
        end_of_season: int | None,
    ) -> None:
        """Save model results to cache with comprehensive error handling.

        Args:
            cache_path: Path to cache directory
            model_name: Name of the model
            predictions_df: Predictions DataFrame
            metrics_by_gauge: Metrics dictionary
            start_of_season: Starting month for seasonal evaluation
            end_of_season: Ending month for seasonal evaluation

        Raises:
            Exception: If saving fails after all retry attempts
        """
        try:
            # Validate data before saving
            if predictions_df.empty:
                self.logger.warning(f"Empty predictions DataFrame for {model_name}, skipping cache save")
                return

            if not metrics_by_gauge:
                self.logger.warning(f"Empty metrics for {model_name}, skipping cache save")
                return

            # Save predictions to parquet with error handling
            predictions_file = cache_path / "predictions" / f"{model_name}.parquet"
            try:
                predictions_df.to_parquet(predictions_file, compression="snappy", index=False)
                self.logger.debug(f"Saved predictions for {model_name} to {predictions_file}")
            except Exception as e:
                raise Exception(f"Failed to save predictions parquet: {e}") from e

            # Convert numpy types to Python types for JSON serialization
            json_safe_metrics = self._convert_to_json_safe(metrics_by_gauge)

            # Save metrics to JSON with error handling
            metrics_file = cache_path / "metrics" / f"{model_name}.json"
            try:
                with open(metrics_file, "w") as f:
                    json.dump(json_safe_metrics, f, indent=2)
                self.logger.debug(f"Saved metrics for {model_name} to {metrics_file}")
            except Exception as e:
                # Clean up predictions file if metrics save failed
                if predictions_file.exists():
                    try:
                        predictions_file.unlink()
                        self.logger.debug("Cleaned up predictions file after metrics save failure")
                    except Exception:
                        pass
                raise Exception(f"Failed to save metrics JSON: {e}") from e

            # Update cache metadata with error handling
            try:
                self._update_cache_metadata_safely(
                    cache_path, model_name, predictions_df, start_of_season, end_of_season
                )
            except Exception as e:
                self.logger.warning(f"Failed to update cache metadata for {model_name}: {e}")
                # Don't raise - the data files are saved successfully

        except Exception as e:
            self.logger.error(f"Failed to save {model_name} to cache: {e}")
            raise

    def _update_cache_metadata_safely(
        self,
        cache_path: Path,
        model_name: str,
        predictions_df: pd.DataFrame,
        start_of_season: int | None,
        end_of_season: int | None,
    ) -> None:
        """Update cache metadata with comprehensive error handling.

        Args:
            cache_path: Path to cache directory
            model_name: Name of the model
            predictions_df: Predictions DataFrame for statistics
            start_of_season: Starting month for seasonal evaluation
            end_of_season: Ending month for seasonal evaluation
        """
        metadata_file = cache_path / "cache_metadata.json"

        try:
            # Load existing metadata or create new with error handling
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    # Validate existing metadata structure
                    if not isinstance(metadata, dict):
                        self.logger.warning("Invalid metadata structure, creating new metadata")
                        metadata = {}

                except (json.JSONDecodeError, PermissionError) as e:
                    self.logger.warning(f"Failed to load existing metadata: {e}. Creating new metadata.")
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

            # Calculate file size safely
            predictions_file = cache_path / "predictions" / f"{model_name}.parquet"
            try:
                file_size_mb = predictions_file.stat().st_size / 1024 / 1024 if predictions_file.exists() else 0
            except OSError:
                file_size_mb = 0
                self.logger.warning(f"Could not calculate file size for {predictions_file}")

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

            # Save updated metadata with atomic write
            temp_file = metadata_file.with_suffix(".tmp")
            try:
                with open(temp_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Atomic rename
                temp_file.replace(metadata_file)
                self.logger.debug(f"Updated cache metadata for {model_name}")

            except Exception as e:
                # Clean up temp file
                if temp_file.exists():
                    with contextlib.suppress(Exception):
                        temp_file.unlink()
                raise e

        except Exception as e:
            self.logger.error(f"Failed to update cache metadata: {e}")
            raise

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
            raise Exception(f"Failed to create cache directory {cache_path}: {e}") from e

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
