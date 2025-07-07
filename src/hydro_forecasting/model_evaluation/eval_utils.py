"""Utility functions for model evaluation and performance analysis."""

from typing import Any

import numpy as np


def _parse_model_results(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Parse model results dictionary to extract architectures and variants.

    Args:
        results: Dictionary from TSForecastEvaluator with model results

    Returns:
        Nested dictionary: {architecture: {variant: model_result}}
    """
    model_data = {}
    for model_name, model_result in results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][variant] = model_result

    return model_data


def extract_metric_values(
    metrics_by_gauge: dict[str, dict[int, dict[str, float]]],
    metric: str,
    horizons: list[int] | None = None,
) -> dict[int, list[float]]:
    """
    Extract metric values for specific horizons across all basins.

    Args:
        metrics_by_gauge: Nested dict with structure {gauge_id: {horizon: {metric: value}}}
        metric: Name of the metric to extract
        horizons: List of horizons to extract. If None, extracts all horizons.

    Returns:
        Dictionary mapping horizon to list of metric values across basins
    """
    horizon_values = {}

    for gauge_data in metrics_by_gauge.values():
        for horizon, horizon_metrics in gauge_data.items():
            # Skip if we're filtering horizons and this isn't in the list
            if horizons is not None and horizon not in horizons:
                continue

            if metric in horizon_metrics:
                value = horizon_metrics[metric]
                if not np.isnan(value):
                    if horizon not in horizon_values:
                        horizon_values[horizon] = []
                    horizon_values[horizon].append(value)

    return horizon_values


def calculate_metric_statistics(values: list[float]) -> dict[str, float]:
    """
    Calculate statistics for a list of metric values.

    Args:
        values: List of metric values

    Returns:
        Dictionary with 'median', 'std', 'mean', 'min', 'max', 'count'
    """
    if not values:
        return {
            "median": np.nan,
            "std": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": 0,
        }

    values_array = np.array(values)
    return {
        "median": np.median(values_array),
        "std": np.std(values_array),
        "mean": np.mean(values_array),
        "min": np.min(values_array),
        "max": np.max(values_array),
        "count": len(values_array),
    }
