from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_horizon_performance_bars(
    results: dict[str, Any],
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
        results: Dictionary from TSForecastEvaluator with model results
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
    # FIX: Renamed loop variable from 'results' to 'model_result' to avoid shadowing
    for model_name, model_result in results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][variant] = model_result

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
                    if horizon in basin_data and metric in basin_data[horizon]:
                        value = basin_data[horizon][metric]
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
    bar_width = 0.18
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
                        color="darkgreen" if (pct_diff >= 0) == positive_is_better else "darkred",
                        fontsize=8,  # Reduced font size by 2 points from default (~10)
                    )

    # Add dummy model horizontal line if specified
    if dummy_model is not None and dummy_model in results:
        dummy_results = results[dummy_model]
        dummy_metrics_by_gauge = dummy_results["metrics_by_gauge"]

        # Extract performance values for the dummy model
        dummy_values = []
        for basin_data in dummy_metrics_by_gauge.values():
            if horizon in basin_data and metric in basin_data[horizon]:
                value = basin_data[horizon][metric]
                if not np.isnan(value):
                    dummy_values.append(value)

        # Draw horizontal line if we have valid data
        if dummy_values:
            dummy_median = float(np.median(dummy_values))
            ax.axhline(
                y=dummy_median,
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Baseline",
                zorder=10,
            )

    # Customize plot
    ax.set_ylabel(f"{metric.upper()} Value")

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels([arch.upper() for arch in architectures])

    # Add grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(variants),
        frameon=False,
        fancybox=True,
        shadow=False,
        facecolor="white",
        edgecolor="gray",
    )

    # Adjust layout

    return fig, ax


def plot_basin_performance_scatter(
    results: dict[str, Any],
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
        results: Dictionary from TSForecastEvaluator with model results
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
    # FIX: Renamed loop variable from 'results' to 'model_result' to avoid shadowing
    for model_name, model_result in results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        pattern = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][pattern] = model_result

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
                horizon in benchmark_data
                and metric in benchmark_data[horizon]
                and horizon in challenger_data
                and metric in challenger_data[horizon]
            ):
                benchmark_val = benchmark_data[horizon][metric]
                challenger_val = challenger_data[horizon][metric]

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
    ax.set_xlabel(f"{metric.upper()} - {benchmark_pattern.capitalize()} Models")
    ax.set_ylabel(f"Δ{metric.upper()} ({challenger_pattern.capitalize()} - {benchmark_pattern.capitalize()})")

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

    return fig, ax


def plot_model_cdf_grid(
    results: dict[str, Any],
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
        results: Dictionary from TSForecastEvaluator with model results
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
    # FIX: Renamed loop variable from 'results' to 'model_result' to avoid shadowing
    for model_name, model_result in results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][variant] = model_result

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
                    if horizon in basin_data and metric in basin_data[horizon]:
                        val = basin_data[horizon][metric]
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
                ax.set_title(arch.upper())
            if j == 0:
                ax.set_ylabel(f"{horizon} Day - CDF")
            if i == len(horizons) - 1:
                ax.set_xlabel(metric.upper())

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
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    return fig, axes


def plot_horizon_performance_boxplots(
    results: dict[str, Any],
    horizon: int,
    metric: str = "NSE",
    architectures: list[str] | None = None,
    variants: list[str] | None = None,
    colors: dict[str, str] | None = None,
    dummy_model: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a boxplot chart showing model performance for a specific horizon.

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        horizon: Forecast horizon to plot (e.g., 5, 10)
        metric: Performance metric to plot (e.g., "NSE", "RMSE")
        architectures: List of architectures to include (auto-detected if None)
        variants: List of variants to include (auto-detected if None)
        colors: Dictionary mapping architecture to color (default colors if None)
        dummy_model: Name of dummy/baseline model to show as boxplot (optional)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)

    Returns:
        Tuple of (figure, axes) objects
    """

    # Parse model names to extract architectures and variants
    model_data = {}
    # FIX: Renamed loop variable from 'results' to 'model_result' to avoid shadowing
    for model_name, model_result in results.items():
        if "_" not in model_name:
            continue

        parts = model_name.split("_", 1)
        arch = parts[0]
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}
        model_data[arch][variant] = model_result

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

                # Extract all metric values for this horizon across all basins
                values = []
                for basin_data in metrics_by_gauge.values():
                    if horizon in basin_data and metric in basin_data[horizon]:
                        value = basin_data[horizon][metric]
                        if not np.isnan(value):
                            values.append(value)

                if values:
                    performance_data[arch][variant] = values

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for boxplots
    boxplot_data = []
    boxplot_labels = []
    boxplot_colors = []
    box_positions = []

    current_position = 1

    # Add dummy model boxplot first if specified
    if dummy_model is not None and dummy_model in results:
        dummy_results = results[dummy_model]
        dummy_metrics_by_gauge = dummy_results["metrics_by_gauge"]

        # Extract performance values for the dummy model
        dummy_values = []
        for basin_data in dummy_metrics_by_gauge.values():
            if horizon in basin_data and metric in basin_data[horizon]:
                value = basin_data[horizon][metric]
                if not np.isnan(value):
                    dummy_values.append(value)

        if dummy_values:
            boxplot_data.append(dummy_values)
            boxplot_labels.append("Dummy")
            boxplot_colors.append("lightgray")
            box_positions.append(current_position)
            current_position += 1.5  # Add gap after dummy model

    # Add model boxplots grouped by architecture and variant
    n_variants = len(variants)
    for arch_idx, arch in enumerate(architectures):
        arch_start_position = current_position

        for variant_idx, variant in enumerate(variants):
            if arch in performance_data and variant in performance_data[arch]:
                boxplot_data.append(performance_data[arch][variant])
                boxplot_labels.append(f"{arch.upper()}\n{variant.capitalize()}")
                boxplot_colors.append(colors[arch])
                box_positions.append(current_position)
                current_position += 1

        # Add gap between architectures
        if arch_idx < len(architectures) - 1:
            current_position += 0.5

    # Create boxplots
    if boxplot_data:
        bp = ax.boxplot(
            boxplot_data,
            positions=box_positions,
            patch_artist=True,
            notch=False,
            widths=0.6,
        )

        # Customize boxplot colors
        for patch, color in zip(bp["boxes"], boxplot_colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1)

        # Customize other boxplot elements
        for element in ["whiskers", "fliers", "medians", "caps"]:
            plt.setp(bp[element], color="black", linewidth=1)

    # Customize plot
    ax.set_ylabel(f"{metric.upper()} Value")

    # Set x-axis
    ax.set_xticks(box_positions)
    ax.set_xticklabels(boxplot_labels)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Set title if provided
    if title:
        ax.set_title(title, pad=20)
    else:
        ax.set_title(f"{metric.upper()} Performance at {horizon}-Day Horizon", pad=20)

    # Create legend for architectures (excluding dummy)
    legend_handles = []
    legend_labels = []

    for arch in architectures:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[arch], alpha=0.7, edgecolor="black"))
        legend_labels.append(arch.upper())

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(architectures),
            frameon=False,
            title="Architectures",
        )

    return fig, ax


def plot_rolling_forecast(
    results: dict[str, Any],
    model_name: str,
    gauge_id: str,
    horizon: int,
    figsize: tuple[int, int] = (15, 7),
    title: str | None = None,
    color_scheme: dict[str, str] | None = None,
    start_of_season: int | None = None,
    end_of_season: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a time-series plot comparing observed values against model's rolling forecasts.

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        model_name: Name of the model to plot (must exist in results)
        gauge_id: Identifier for the gauge/basin to plot
        horizon: Forecast horizon to plot (e.g., 1, 5, 10)
        figsize: Figure size as (width, height)
        title: Custom plot title (auto-generated if None)
        color_scheme: Dictionary mapping 'observed', 'predicted', and 'season' to colors
        start_of_season: Start month of seasonal period (1-12, optional)
        end_of_season: End month of seasonal period (1-12, optional)

    Returns:
        Tuple of (figure, axes) objects

    Raises:
        ValueError: If model_name not found or no data for specified gauge/horizon
    """

    # Check if model exists in results
    if model_name not in results:
        available_models = list(results.keys())
        raise ValueError(f"Model '{model_name}' not found in results. Available models: {available_models}")

    # Extract predictions DataFrame for the specified model
    model_results = results[model_name]
    if "predictions_df" not in model_results:
        raise ValueError(f"No predictions_df found for model '{model_name}'")

    predictions_df = model_results["predictions_df"]

    # Filter for the specific gauge and horizon
    filtered_df = predictions_df[
        (predictions_df["gauge_id"] == gauge_id) & (predictions_df["horizon"] == horizon)
    ].copy()

    # Check if we have data after filtering
    if filtered_df.empty:
        available_gauges = predictions_df["gauge_id"].unique()
        available_horizons = predictions_df["horizon"].unique()
        raise ValueError(
            f"No data found for gauge '{gauge_id}' and horizon {horizon} "
            f"in model '{model_name}'. Available gauges: {list(available_gauges)}, "
            f"Available horizons: {list(available_horizons)}"
        )

    # Sort by date for proper time series plotting
    filtered_df = filtered_df.sort_values("date")

    # Set default color scheme if not provided
    if color_scheme is None:
        color_scheme: dict[str, str] = {"observed": "black", "predicted": "red", "season": "#d3d3d3"}
    else:
        # Ensure season color exists in provided color scheme
        if "season" not in color_scheme:
            color_scheme["season"] = "#d3d3d3"

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Add seasonal background shading if both start and end months are provided
    if start_of_season is not None and end_of_season is not None:
        # Get unique years from the data
        unique_years = sorted(filtered_df["date"].dt.year.unique())
        season_labeled = False  # Flag to label only the first span

        for year in unique_years:
            # Define the start date for the current year's season
            season_start_date = mdates.datetime.datetime(year, start_of_season, 1)

            # Determine the year for the end of the season
            # If the season crosses a year boundary (e.g., Dec-Feb), the end year is the next year
            end_year = year + 1 if end_of_season < start_of_season else year
            season_end_date = mdates.datetime.datetime(end_year, end_of_season, 1)

            # Draw the shaded span
            ax.axvspan(
                season_start_date,
                season_end_date,
                color=color_scheme["season"],
                alpha=0.2,
                zorder=0,
                label="Season" if not season_labeled else "",
            )
            season_labeled = True

    # Plot observed and predicted values
    ax.plot(
        filtered_df["date"],
        filtered_df["observed"],
        color=color_scheme.get("observed", "black"),
        linewidth=1.5,
        label="Observed",
        alpha=0.8,
    )

    ax.plot(
        filtered_df["date"],
        filtered_df["predicted"],
        color=color_scheme.get("predicted", "red"),
        linewidth=1.5,
        label="Predicted",
        alpha=0.8,
    )

    # Set title
    if title is None:
        title = f"{horizon}-Day Rolling Forecast for {model_name} at Gauge {gauge_id}"
    ax.set_title(title, pad=20)

    ax.set_xlabel("")
    ax.set_ylabel("Value")

    ax.legend(loc="best", frameon=False, fancybox=True, shadow=False)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    return fig, ax


def plot_performance_vs_test_length(
    results: dict[str, Any],
    model_name: str,
    horizon: int,
    metric: str = "NSE",
    figsize: tuple[int, int] = (12, 7),
    add_trendline: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a scatter plot showing model performance against the length of the testing period for each basin.

    This function analyzes the relationship between test period duration and model performance
    across different basins. It can optionally include a linear trendline with R-squared statistics
    to quantify the relationship between test length and performance.

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        model_name: Name of the model to analyze (must exist in results)
        horizon: Forecast horizon to analyze (e.g., 1, 5, 10)
        metric: Performance metric to plot (e.g., "NSE", "RMSE")
        figsize: Figure size as (width, height)
        add_trendline: Whether to add a linear trendline with R-squared statistics

    Returns:
        Tuple of (figure, axes) objects

    Raises:
        ValueError: If model_name not found in results, or if required data is missing
    """

    # Input validation - check if model exists
    if model_name not in results:
        available_models = list(results.keys())
        raise ValueError(f"Model '{model_name}' not found in results. Available models: {available_models}")

    # Extract model results
    model_results = results[model_name]

    # Check for required data structures
    if "predictions_df" not in model_results:
        raise ValueError(f"No predictions_df found for model '{model_name}'")
    if "metrics_by_gauge" not in model_results:
        raise ValueError(f"No metrics_by_gauge found for model '{model_name}'")

    predictions_df = model_results["predictions_df"]
    metrics_by_gauge = model_results["metrics_by_gauge"]

    # Calculate test period length for each basin at the specific horizon
    # Filter predictions to only include the specified horizon
    horizon_filtered_df = predictions_df[predictions_df["horizon"] == horizon]

    # Use pandas groupby for efficient aggregation
    test_lengths = (
        horizon_filtered_df.groupby("gauge_id")["observed"].apply(lambda x: x.notna().sum() / 365.25).to_dict()
    )

    # Collate plotting data
    plot_data = []
    for gauge_id, gauge_metrics in metrics_by_gauge.items():
        # Check if we have both test length and metric data for this gauge
        if gauge_id not in test_lengths:
            continue

        if horizon not in gauge_metrics or metric not in gauge_metrics[horizon]:
            continue

        metric_value = gauge_metrics[horizon][metric]
        test_length = test_lengths[gauge_id]

        # Only include valid (non-NaN) metric values
        if not np.isnan(metric_value):
            plot_data.append({"gauge_id": gauge_id, "test_length_years": test_length, "metric_value": metric_value})

    # Check if we have any valid data to plot
    if not plot_data:
        raise ValueError(
            f"No valid data found for plotting. Check that gauge IDs match between predictions_df and metrics_by_gauge for horizon {horizon} and metric {metric}"
        )

    # Convert to DataFrame for easier handling
    plot_df = pd.DataFrame(plot_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    ax.scatter(
        plot_df["test_length_years"],
        plot_df["metric_value"],
        color="#4682B4",
        s=80,
        alpha=0.7,
        edgecolors="darkblue",
        linewidth=0.5,
        label="Basin Performance",
    )

    # Add trendline if requested
    if add_trendline and len(plot_df) > 1:
        x_values = plot_df["test_length_years"].values
        y_values = plot_df["metric_value"].values

        # Use scipy for cleaner trendline calculation
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        r_squared = r_value**2

        # Plot trendline
        x_trend = np.linspace(x_values.min(), x_values.max(), 100)
        y_trend = slope * x_trend + intercept

        ax.plot(
            x_trend,
            y_trend,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Trend (y={slope:.3f}x + {intercept:.3f}, R²={r_squared:.3f})",
        )

    # Customize plot aesthetics
    ax.set_xlabel("Length of Test Period (Years)")
    ax.set_ylabel(f"{metric.upper()}")
    ax.set_title(f"{metric.upper()} vs. Test Period Length for {model_name} at {horizon}-Day Horizon", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="best", frameon=False, fancybox=True, shadow=False)

    return fig, ax
