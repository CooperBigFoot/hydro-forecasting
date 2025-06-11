from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# TODO: Fix plots to work with new TSForecastEvaluator results format


def plot_horizon_performance_bars(
    seasonal_results: dict[str, Any],
    horizon: int,
    metric: str = "NSE",
    architectures: list[str] | None = None,
    variants: list[str] | None = None,
    colors: dict[str, str] | None = None,
    dummy_model: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    with_whiskers: bool = True,
    positive_is_better: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a bar chart showing model performance for a specific horizon.

    Args:
        seasonal_results: Dictionary from TSForecastEvaluator with model results
        horizon: Forecast horizon to plot (e.g., 5, 10)
        metric: Performance metric to plot (e.g., "NSE", "RMSE")
        architectures: List of architectures to include (auto-detected if None)
        variants: List of variants to include (auto-detected if None)
        colors: Dictionary mapping architecture to color (default colors if None)
        dummy_model: Name of dummy/baseline model to show as horizontal line (optional)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        with_whiskers: Whether to show error bars (whiskers). If False, shows % difference from benchmark
        positive_is_better: Whether positive differences indicate improvement (True for NSE, False for RMSE)

    Returns:
        Tuple of (figure, axes) objects
    """

    # Parse model names to extract architectures and variants
    model_data = {}
    for model_name, results in seasonal_results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][variant] = results

    # Auto-detect architectures and variants if not provided
    if architectures is None:
        architectures = sorted(model_data.keys())
    if variants is None:
        variants = sorted({v for arch_variants in model_data.values() for v in arch_variants})

    # Default colors
    if colors is None:
        default_colors = ["#4682B4", "#CD5C5C", "#009E73", "#9370DB", "#FF8C00", "#8B4513"]
        colors = {arch: default_colors[i % len(default_colors)] for i, arch in enumerate(architectures)}

    # Extract performance data
    performance_data = {}
    for arch in architectures:
        performance_data[arch] = {}
        for variant in variants:
            if arch in model_data and variant in model_data[arch]:
                metrics_by_gauge = model_data[arch][variant]["metrics_by_gauge"]

                # Extract metric values for this horizon across all basins
                values = []
                for basin_data in metrics_by_gauge.values():
                    if f"horizon_{horizon}" in basin_data and metric in basin_data[f"horizon_{horizon}"]:
                        value = basin_data[f"horizon_{horizon}"][metric]
                        if not np.isnan(value):
                            values.append(value)

                if values:
                    performance_data[arch][variant] = {
                        "median": np.median(values),
                        "std": np.std(values),
                        "count": len(values),
                    }

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define patterns for variants
    patterns = {
        "benchmark": None,  # Solid
        "pretrained": "///",  # Diagonal lines
        "finetuned": "...",  # Dots
    }
    # Add more patterns if needed
    if len(variants) > 3:
        additional_patterns = ["+++", "xxx", "|||", "---", "ooo"]
        for i, variant in enumerate(variants[3:]):
            patterns[variant] = additional_patterns[i % len(additional_patterns)]

    # Plot parameters
    n_architectures = len(architectures)
    n_variants = len(variants)
    bar_width = 0.15
    group_width = n_variants * bar_width
    x_positions = np.arange(n_architectures)

    # Plot bars
    benchmark_values = {}  # Store benchmark values for each architecture

    for variant_idx, variant in enumerate(variants):
        x_offset = (variant_idx - (n_variants - 1) / 2) * bar_width

        medians = []
        stds = []
        colors_list = []

        for arch_idx, arch in enumerate(architectures):
            if arch in performance_data and variant in performance_data[arch]:
                median_val = performance_data[arch][variant]["median"]
                medians.append(median_val)
                stds.append(performance_data[arch][variant]["std"])
                colors_list.append(colors[arch])

                # Store benchmark values for percentage calculation
                if variant == "benchmark":
                    benchmark_values[arch] = median_val
            else:
                medians.append(0)
                stds.append(0)
                colors_list.append("#CCCCCC")  # Gray for missing data

        # Create bars
        bars = ax.bar(
            x_positions + x_offset,
            medians,
            bar_width,
            yerr=stds if with_whiskers else None,
            capsize=3 if with_whiskers else 0,
            color=colors_list,
            hatch=patterns.get(variant),
            edgecolor="black",
            linewidth=0.8,
            label=variant.capitalize(),
            alpha=0.8 if patterns.get(variant) else 1.0,
        )

        # Add percentage difference annotations when whiskers are disabled
        if not with_whiskers and variant != "benchmark":
            for arch_idx, (arch, bar) in enumerate(zip(architectures, bars, strict=False)):
                if (
                    arch in performance_data
                    and variant in performance_data[arch]
                    and arch in benchmark_values
                    and benchmark_values[arch] != 0
                ):
                    current_val = performance_data[arch][variant]["median"]
                    benchmark_val = benchmark_values[arch]

                    # Calculate percentage difference
                    pct_diff = ((current_val - benchmark_val) / abs(benchmark_val)) * 100

                    # Position text above the bar
                    bar_height = bar.get_height()
                    text_y = bar_height + ax.get_ylim()[1] * 0.02  # Small offset above bar

                    # Format percentage text
                    sign = "+" if pct_diff >= 0 else ""
                    pct_text = f"{sign}{pct_diff:.1f}%"

                    # Add text annotation
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        text_y,
                        pct_text,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="darkgreen" if (pct_diff >= 0) == positive_is_better else "darkred",
                    )

    # Customize plot
    ax.set_ylabel(f"{metric} Value")

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels([arch.upper() for arch in architectures], fontsize=11)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=len(variants),
        frameon=False,
        fancybox=True,
        shadow=False,
        facecolor="white",
        edgecolor="gray",
    )

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_basin_performance_scatter(
    seasonal_results: dict[str, Any],
    benchmark_pattern: str,
    challenger_pattern: str,
    horizon: int,
    metric: str = "NSE",
    architectures: list[str] | None = None,
    colors: dict[str, str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    debug: bool = False,
    significance_band_width: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a scatter plot comparing basin-level performance between benchmark and challenger models.

    Args:
        seasonal_results: Dictionary from TSForecastEvaluator with model results
        benchmark_pattern: Pattern to identify benchmark models (e.g., "benchmark", "pretrained")
        challenger_pattern: Pattern to identify challenger models (e.g., "finetuned", "pretrained")
        horizon: Forecast horizon to plot (e.g., 5, 10)
        metric: Performance metric to plot (e.g., "NSE", "RMSE")
        architectures: List of architectures to include (auto-detected if None)
        colors: Dictionary mapping architecture to color (default colors if None)
        figsize: Figure size as (width, height)
        debug: Whether to show gauge IDs as labels on scatter points
        significance_band_width: Multiplier for standard error to define significance band width (e.g., 1.0 for ±1 SE, 1.96 for ±95% CI)

    Returns:
        Tuple of (figure, axes) objects
    """

    # Parse model names to extract architectures and patterns
    model_data = {}
    for model_name, results in seasonal_results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        pattern = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][pattern] = results

    # Auto-detect architectures if not provided
    if architectures is None:
        architectures = sorted(model_data.keys())

    # Default colors
    if colors is None:
        default_colors = ["#4682B4", "#CD5C5C", "#009E73", "#9370DB", "#FF8C00", "#8B4513"]
        colors = {arch: default_colors[i % len(default_colors)] for i, arch in enumerate(architectures)}

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Collect all scatter plot data
    all_benchmark_values = []
    all_delta_values = []

    for arch in architectures:
        if arch not in model_data:
            continue

        # Check if both benchmark and challenger patterns exist for this architecture
        if benchmark_pattern not in model_data[arch] or challenger_pattern not in model_data[arch]:
            continue

        benchmark_metrics_by_gauge = model_data[arch][benchmark_pattern]["metrics_by_gauge"]
        challenger_metrics_by_gauge = model_data[arch][challenger_pattern]["metrics_by_gauge"]

        # Find common basins between benchmark and challenger
        benchmark_basins = set(benchmark_metrics_by_gauge.keys())
        challenger_basins = set(challenger_metrics_by_gauge.keys())
        common_basins = benchmark_basins.intersection(challenger_basins)

        if not common_basins:
            continue

        # Extract performance values for common basins
        benchmark_values = []
        delta_values = []
        basin_ids = []  # For debug labels

        for basin_id in common_basins:
            # Check if horizon and metric exist for both models
            benchmark_data = benchmark_metrics_by_gauge[basin_id]
            challenger_data = challenger_metrics_by_gauge[basin_id]

            if (
                f"horizon_{horizon}" in benchmark_data
                and metric in benchmark_data[f"horizon_{horizon}"]
                and f"horizon_{horizon}" in challenger_data
                and metric in challenger_data[f"horizon_{horizon}"]
            ):
                benchmark_val = benchmark_data[f"horizon_{horizon}"][metric]
                challenger_val = challenger_data[f"horizon_{horizon}"][metric]

                # Skip if either value is NaN
                if not (np.isnan(benchmark_val) or np.isnan(challenger_val)):
                    delta_val = challenger_val - benchmark_val
                    benchmark_values.append(benchmark_val)
                    delta_values.append(delta_val)
                    basin_ids.append(basin_id)

        if benchmark_values and delta_values:
            # Create scatter plot for this architecture
            ax.scatter(
                benchmark_values,
                delta_values,
                color=colors[arch],
                label=arch.upper(),
                s=80,
                edgecolors=colors[arch],
                linewidth=0.5,
            )

            # Add debug labels if requested
            if debug:
                for x, y, basin_id in zip(benchmark_values, delta_values, basin_ids, strict=True):
                    ax.annotate(
                        basin_id,
                        (x, y),
                        xytext=(5, 5),  # Offset in points
                        textcoords="offset points",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                    )

            # Collect values for overall plot limits
            all_benchmark_values.extend(benchmark_values)
            all_delta_values.extend(delta_values)

    # Calculate standard error of differences for significance band
    if all_delta_values:
        delta_std_error = np.std(all_delta_values) / np.sqrt(len(all_delta_values))
        significance_threshold = delta_std_error * significance_band_width
    else:
        significance_threshold = 0

    # Customize plot
    ax.set_xlabel(f"{metric} - {benchmark_pattern.capitalize()} Models")
    ax.set_ylabel(f"Δ{metric} ({challenger_pattern.capitalize()} - {benchmark_pattern.capitalize()})")

    # Set limits with margin
    if all_benchmark_values and all_delta_values:
        # X-axis limits
        x_min, x_max = min(all_benchmark_values), max(all_benchmark_values)
        x_margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

        # Y-axis limits
        y_min, y_max = min(all_delta_values), max(all_delta_values)
        y_margin = max(abs(y_min), abs(y_max)) * 0.05
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add shaded area for no significant change (±significance_band_width standard errors)
        x_range = np.linspace(x_min - x_margin, x_max + x_margin, 100)
        ax.fill_between(
            x_range,
            -significance_threshold,
            significance_threshold,
            color="lightgray",
            alpha=0.3,
            label=f"No Significant Change (±{significance_band_width:.1f} SE)",
        )

        # Add horizontal line at y=0 (no improvement/degradation)
        ax.axhline(y=0, color="darkgray", linestyle="--", alpha=0.7, linewidth=1.5, label="No Change")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    if len(architectures) > 1:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(architectures),
            frameon=False,
            fancybox=True,
            shadow=False,
            facecolor="white",
            edgecolor="gray",
        )
    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_model_cdf_grid(
    seasonal_results: dict[str, Any],
    horizons: list[int] | None = None,
    metric: str = "NSE",
    architectures: list[str] | None = None,
    variants: list[str] | None = None,
    colors: dict[str, str] | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot a grid of CDF curves for different models and prediction horizons.

    Args:
        seasonal_results: Dictionary from TSForecastEvaluator with model results
        horizons: List of forecast horizons to plot (defaults to [1, 5, 10])
        metric: Performance metric to plot (default: "NSE")
        architectures: List of architectures to include (auto-detected if None)
        variants: List of variants to include (auto-detected if None)
        colors: Dictionary mapping architecture to color (default colors if None)
        figsize: Figure size as (width, height)

    Returns:
        Tuple of (figure, list of axes)
    """

    # Parse model names to extract architectures and variants
    model_data = {}
    for model_name, results in seasonal_results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][variant] = results

    # Auto-detect architectures and variants if not provided
    if architectures is None:
        architectures = sorted(model_data.keys())
    if variants is None:
        variants = sorted({v for arch_variants in model_data.values() for v in arch_variants})
    if horizons is None:
        horizons = [1, 5, 10]

    # Default colors
    if colors is None:
        default_colors = ["#4682B4", "#CD5C5C", "#009E73", "#9370DB", "#FF8C00", "#8B4513"]
        colors = {arch: default_colors[i % len(default_colors)] for i, arch in enumerate(architectures)}

    # Define line styles for up to 5 variants
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, densely dashdotdotted
    variant_styles = {variant: line_styles[i % len(line_styles)] for i, variant in enumerate(variants)}

    # Define x-axis limits for each horizon
    horizon_xlims = {1: (0.5, 1.0), 5: (0.5, 1.0), 10: (0.5, 1.0), 15: (0.5, 1.0), 20: (0.5, 1.0), 30: (0.5, 1.0)}

    # Create figure and subplots
    fig, axes = plt.subplots(len(horizons), len(architectures), figsize=figsize)

    # Handle case where there's only one horizon or one architecture
    if len(horizons) == 1 and len(architectures) == 1:
        axes = np.array([[axes]])
    elif len(horizons) == 1:
        axes = axes.reshape(1, -1)
    elif len(architectures) == 1:
        axes = axes.reshape(-1, 1)

    # Track median values for legend
    median_lines_added = False

    # Loop through each row (horizon)
    for i, horizon in enumerate(horizons):
        # Loop through each column (architecture)
        for j, arch in enumerate(architectures):
            ax = axes[i, j]

            if arch not in model_data:
                # No data for this architecture, skip
                ax.set_visible(False)
                continue

            # For each variant of this architecture
            for variant in variants:
                if variant not in model_data[arch]:
                    continue

                # Extract metric values for the specific horizon
                metric_values = []

                # Get the metrics by gauge for this model
                metrics_by_gauge = model_data[arch][variant]["metrics_by_gauge"]

                for basin_id, basin_data in metrics_by_gauge.items():
                    if f"horizon_{horizon}" in basin_data and metric in basin_data[f"horizon_{horizon}"]:
                        val = basin_data[f"horizon_{horizon}"][metric]
                        # Convert numpy types to Python float if needed
                        if hasattr(val, "item"):
                            val = val.item()
                        # Only add non-NaN values
                        if not np.isnan(val):
                            metric_values.append(val)

                # Plot CDF if we have data
                if metric_values:
                    # Sort values for CDF
                    metric_values.sort()

                    # Calculate cumulative probabilities
                    prob = np.arange(1, len(metric_values) + 1) / len(metric_values)

                    # Plot CDF curve
                    ax.plot(
                        metric_values,
                        prob,
                        label=variant.capitalize(),
                        color=colors[arch],
                        linestyle=variant_styles[variant],
                        linewidth=1.8,
                    )

                    # Calculate and plot median
                    median_val = np.median(metric_values)
                    ax.axvline(
                        x=median_val, color=colors[arch], linestyle=variant_styles[variant], linewidth=1.5, alpha=0.7
                    )

            # Set standardized x-axis limits based on horizon
            if horizon in horizon_xlims:
                ax.set_xlim(horizon_xlims[horizon])

            # Ensure key values are included in the x-ticks
            current_ticks = list(ax.get_xticks())
            if 1.0 not in current_ticks:
                current_ticks.append(1.0)
                ax.set_xticks(sorted(current_ticks))

            # Add titles and labels
            if i == 0:
                ax.set_title(arch.upper(), fontsize=14)
            if j == 0:
                ax.set_ylabel(f"{horizon} Day - CDF", fontsize=12)
            if i == len(horizons) - 1:
                ax.set_xlabel(metric, fontsize=12)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle="--")

            # Set y-axis to show values from 0 to 1
            ax.set_ylim(0, 1)

    # Create legend with variants (line styles)
    variant_handles = []
    variant_labels = []

    for variant in variants:
        variant_handles.append(plt.Line2D([0], [0], color="black", linestyle=variant_styles[variant], linewidth=1.8))
        variant_labels.append(variant.capitalize())

    # Add median line to legend
    variant_handles.append(plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, alpha=0.5))
    variant_labels.append("Median")

    # Add legend
    fig.legend(
        handles=variant_handles,
        labels=variant_labels,
        loc="lower center",
        ncol=len(variants) + 1,  # +1 for median line
        bbox_to_anchor=(0.5, 0.02),
        fontsize=12,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    return fig, axes
