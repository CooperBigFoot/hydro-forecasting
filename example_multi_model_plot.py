#!/usr/bin/env python3
"""
Example of using the updated plot_rolling_forecast function with multiple models.
"""

from hydro_forecasting.model_evaluation.visualization import plot_rolling_forecast
import matplotlib.pyplot as plt
import seaborn as sns

# Example usage with your notebook setup:

# Single model (backward compatible)
fig, ax = plot_rolling_forecast(
    results,
    model_names="tsmixer_benchmark",  # Can still pass a single string
    gauge_id="CA_15102", 
    horizon=10,
    figsize=(10, 5),
    colors={"tsmixer_benchmark": "#009E73"},  # Model-specific color
    observed_color="#2E4057",
)

# Multiple models with harmonized date range
fig, ax = plot_rolling_forecast(
    results,
    model_names=["tsmixer_benchmark", "tft_benchmark", "ealstm_benchmark", "tide_benchmark"],
    gauge_id="CA_15102",
    horizon=10, 
    figsize=(12, 6),
    colors={
        "tsmixer_benchmark": "#009E73",
        "tft_benchmark": "#9370DB",
        "ealstm_benchmark": "#CD5C5C", 
        "tide_benchmark": "#4682B4"
    },
    observed_color="#2E4057",
)

sns.despine()
ax.set_xlabel("")
ax.set_ylabel("Streamflow (mm/day)")
plt.show()

# The function will automatically:
# 1. Find the common date range across all models
# 2. Plot observed values once
# 3. Plot predicted values for each model with specified colors
# 4. Create appropriate legend entries