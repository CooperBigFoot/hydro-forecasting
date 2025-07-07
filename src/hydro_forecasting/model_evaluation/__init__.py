"""Model evaluation module for hydro forecasting."""

from .evaluators import TSForecastEvaluator
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
from .performance_evaluation import generate_performance_summary, print_performance_summary
from .visualization import (
    generate_brightness_gradient,
    plot_basin_performance_scatter,
    plot_horizon_performance_bars,
    plot_model_cdf_grid,
    remaining_skill_captured_vs_horizon,
)

__all__ = [
    # Evaluators
    "TSForecastEvaluator",
    # Metrics
    "calculate_atpe",
    "calculate_kge",
    "calculate_mae",
    "calculate_mse",
    "calculate_nse",
    "calculate_pbias",
    "calculate_pearson_r",
    "calculate_rmse",
    # Performance evaluation
    "generate_performance_summary",
    "print_performance_summary",
    # Visualization
    "generate_brightness_gradient",
    "plot_basin_performance_scatter",
    "plot_horizon_performance_bars",
    "plot_model_cdf_grid",
    "remaining_skill_captured_vs_horizon",
]
