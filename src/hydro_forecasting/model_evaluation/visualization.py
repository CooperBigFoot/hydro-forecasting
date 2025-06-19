from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_brightness_gradient(hex_color: str, count: int) -> list[str]:
    """
    Generate a list of colors with increasing brightness from a base hex color.

    Args:
        hex_color: Base hex color code (e.g., '#FF5733' or 'FF5733').
        count: Number of colors to generate in the gradient.

    Returns:
        List of hex color codes, starting with the original color and
        progressively getting brighter.

    Raises:
        ValueError: If hex_color is invalid or count is less than 1.
    """
    if count < 1:
        raise ValueError("Count must be at least 1")

    # Remove '#' if present and validate hex color
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6 or not all(c in "0123456789ABCDEFabcdef" for c in hex_color):
        raise ValueError(f"Invalid hex color: {hex_color}")

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    colors = []

    for i in range(count):
        if i == 0:
            # First element is the original color
            colors.append(f"#{hex_color.upper()}")
        else:
            brightness_factor = (i / (count - 1)) * 0.7 if count > 1 else 0

            # Brighten by interpolating towards white (255, 255, 255)
            new_r = int(r + (255 - r) * brightness_factor)
            new_g = int(g + (255 - g) * brightness_factor)
            new_b = int(b + (255 - b) * brightness_factor)

            # Ensure values stay within 0-255 range
            new_r = min(255, max(0, new_r))
            new_g = min(255, max(0, new_g))
            new_b = min(255, max(0, new_b))

            # Convert back to hex
            hex_result = f"#{new_r:02X}{new_g:02X}{new_b:02X}"
            colors.append(hex_result)

    return colors


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

    # Generate color gradients for each architecture
    arch_gradients = {}
    for arch in architectures:
        arch_gradients[arch] = generate_brightness_gradient(colors[arch], len(variants))

    # Create mapping from variant to gradient index
    variant_to_index = {variant: i for i, variant in enumerate(variants)}

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

                    # Get color from gradient
                    variant_index = variant_to_index[variant]
                    variant_color = arch_gradients[arch][variant_index]

                    # Plot CDF curve
                    ax.plot(
                        metric_values,
                        prob,
                        label=variant.capitalize(),
                        color=variant_color,
                        linestyle=variant_styles[variant],
                        linewidth=1.8,
                    )

                    # Calculate and plot median
                    median_val = np.median(metric_values)
                    ax.axvline(
                        x=median_val, color=variant_color, linestyle=variant_styles[variant], linewidth=1.5, alpha=0.7
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

    # Create legend with variants (line styles and gradient colors)
    variant_handles = []
    variant_labels = []

    # Use sample gradient colors for legend (using first architecture's gradient)
    sample_arch = architectures[0] if architectures else None
    if sample_arch and sample_arch in arch_gradients:
        sample_gradient = arch_gradients[sample_arch]
    else:
        # Fallback to grayscale if no architectures
        sample_gradient = generate_brightness_gradient("#000000", len(variants))

    for i, variant in enumerate(variants):
        variant_handles.append(
            plt.Line2D([0], [0], color=sample_gradient[i], linestyle=variant_styles[variant], linewidth=1.8)
        )
        variant_labels.append(variant.capitalize())

    # Add median line to legend
    variant_handles.append(plt.Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5, alpha=0.7))
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
    horizons: list[int],
    metric: str = "NSE",
    architectures: list[str] | None = None,
    variants: list[str] | None = None,
    colors: dict[str, str] | None = None,
    variant_mapping: dict[str, str] | None = None,
    figsize: tuple[int, int] = (12, 6),
    title: str | None = None,
    show_median_labels: bool = True,
    individual_points: bool = False,
    dummy_model: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a boxplot showing model performance across multiple forecast horizons.

    This function creates a plot where:
    - X-axis: Forecast horizons (1, 5, 10, etc.)
    - Y-axis: Performance metric values
    - Groups: Different model variants (benchmark, pretrained, finetuned)
    - Colors: Different architectures (LSTM, GRU, etc.) with variant-specific brightness

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        horizons: List of forecast horizons to plot (e.g., [1, 5, 10])
        metric: Performance metric to plot (e.g., "NSE", "RMSE")
        architectures: List of architectures to include (auto-detected if None)
        variants: List of variants to include (auto-detected if None)
        colors: Dictionary mapping architecture to base color (default colors if None)
        variant_mapping: Custom display names for variants {"benchmark": "Benchmark Models"}
        figsize: Figure size as (width, height)
        title: Plot title (auto-generated if None)
        show_median_labels: Whether to show median values as text on boxes
        individual_points: Whether to overlay individual basin points
        dummy_model: Name of dummy/baseline model to show as boxplot (optional)

    Returns:
        Tuple of (figure, axes) objects
    """

    # Parse model names to extract architectures and variants
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

    # Auto-detect architectures and variants if not provided
    if architectures is None:
        architectures = sorted(model_data.keys())
    if variants is None:
        variants = sorted({v for arch_variants in model_data.values() for v in arch_variants})

    # Default colors
    if colors is None:
        default_colors = ["#4682B4", "#CD5C5C", "#009E73", "#9370DB", "#FF8C00", "#8B4513"]
        colors = {arch: default_colors[i % len(default_colors)] for i, arch in enumerate(architectures)}

    # Default variant mapping
    if variant_mapping is None:
        variant_mapping = {variant: variant.capitalize() for variant in variants}

    # Extract performance data across all horizons
    performance_data = {}
    for arch in architectures:
        performance_data[arch] = {}
        for variant in variants:
            performance_data[arch][variant] = {}
            if arch in model_data and variant in model_data[arch]:
                metrics_by_gauge = model_data[arch][variant]["metrics_by_gauge"]

                # Extract metric values for each horizon across all basins
                for horizon in horizons:
                    values = []
                    for basin_data in metrics_by_gauge.values():
                        if horizon in basin_data and metric in basin_data[horizon]:
                            value = basin_data[horizon][metric]
                            if not np.isnan(value):
                                values.append(value)

                    if values:
                        performance_data[arch][variant][horizon] = values

    # Extract dummy model data if specified
    dummy_data = {}
    if dummy_model is not None and dummy_model in results:
        dummy_results = results[dummy_model]
        dummy_metrics_by_gauge = dummy_results["metrics_by_gauge"]

        for horizon in horizons:
            dummy_values = []
            for basin_data in dummy_metrics_by_gauge.values():
                if horizon in basin_data and metric in basin_data[horizon]:
                    value = basin_data[horizon][metric]
                    if not np.isnan(value):
                        dummy_values.append(value)

            if dummy_values:
                dummy_data[horizon] = dummy_values

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for seaborn-style grouped boxplots
    plot_data = []

    # Add dummy model data
    if dummy_data:
        for horizon in horizons:
            if horizon in dummy_data:
                for value in dummy_data[horizon]:
                    plot_data.append(
                        {"horizon": horizon, "architecture": "Dummy", "variant": "baseline", "value": value}
                    )

    # Add model data
    for arch in architectures:
        for variant in variants:
            if arch in performance_data and variant in performance_data[arch]:
                for horizon in horizons:
                    if horizon in performance_data[arch][variant]:
                        for value in performance_data[arch][variant][horizon]:
                            plot_data.append(
                                {"horizon": horizon, "architecture": arch, "variant": variant, "value": value}
                            )

    # Create DataFrame for plotting
    df = pd.DataFrame(plot_data)

    if df.empty:
        # Handle empty data case
        ax.text(
            0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray"
        )
        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel(f"{metric.upper()} Value")
        return fig, ax

    # Create color mapping
    color_map = {}

    # Add dummy color if present
    if dummy_data:
        color_map[("Dummy", "baseline")] = "lightgray"

    # Add architecture-variant color combinations
    for arch in architectures:
        if arch in performance_data:
            # Generate brightness gradient for this architecture
            arch_gradients = generate_brightness_gradient(colors[arch], len(variants))

            for variant_idx, variant in enumerate(variants):
                if variant in performance_data[arch]:
                    color_map[(arch, variant)] = arch_gradients[variant_idx]

    # Create custom hue column for color mapping
    df["hue_key"] = list(zip(df["architecture"], df["variant"], strict=False))
    df["color"] = df["hue_key"].map(color_map)

    # Calculate positions for grouped boxplots
    n_models = len([key for key in color_map if key[0] != "Dummy"])
    n_dummy = 1 if dummy_data else 0
    total_models = n_models + n_dummy

    # Make boxes wider and adjust spacing
    if total_models <= 6:
        box_width = 0.12  # Wider boxes for fewer models
    elif total_models <= 12:
        box_width = 0.08  # Medium boxes
    else:
        box_width = 0.06  # Narrower boxes for many models

    # Create boxplots manually for better control
    boxplot_data = []
    positions = []
    colors_list = []
    labels = []

    for h_idx, horizon in enumerate(sorted(horizons)):
        horizon_data = df[df["horizon"] == horizon]

        # Add dummy model first if present
        if dummy_data and horizon in dummy_data:
            dummy_horizon_data = horizon_data[horizon_data["architecture"] == "Dummy"]
            if not dummy_horizon_data.empty:
                pos = h_idx - (total_models - 1) * box_width / 2
                boxplot_data.append(dummy_horizon_data["value"].values)
                positions.append(pos)
                colors_list.append("lightgray")
                labels.append("Dummy")

        # Add architecture-variant combinations
        model_idx = n_dummy
        for arch in architectures:
            for variant in variants:
                model_data = horizon_data[(horizon_data["architecture"] == arch) & (horizon_data["variant"] == variant)]

                if not model_data.empty:
                    pos = h_idx + (model_idx - (total_models - 1) / 2) * box_width
                    boxplot_data.append(model_data["value"].values)
                    positions.append(pos)
                    colors_list.append(color_map.get((arch, variant), "gray"))
                    labels.append(f"{arch}_{variant}")
                    model_idx += 1

    # Create boxplots
    if boxplot_data:
        bp = ax.boxplot(
            boxplot_data,
            positions=positions,
            patch_artist=True,
            widths=box_width * 0.9,  # Use most of the allocated width
            showfliers=not individual_points,
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors_list, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        # Customize other elements
        for element in ["whiskers", "fliers", "medians", "caps"]:
            plt.setp(bp[element], color="black", linewidth=1)

        # Add median labels if requested
        if show_median_labels and len(boxplot_data) <= 20:  # Don't overcrowd
            for i, (data, pos) in enumerate(zip(boxplot_data, positions, strict=False)):
                if len(data) > 0:
                    median_val = np.median(data)
                    ax.text(
                        pos,
                        median_val + ax.get_ylim()[1] * 0.02,
                        f"{median_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                        color="black",
                    )

        # Add individual points if requested
        if individual_points:
            for i, (data, pos) in enumerate(zip(boxplot_data, positions, strict=False)):
                if len(data) > 0:
                    # Add small random jitter for visibility
                    jitter = np.random.normal(0, box_width * 0.1, len(data))
                    ax.scatter(
                        [pos] * len(data) + jitter,
                        data,
                        alpha=0.4,
                        s=15,
                        color=colors_list[i],
                        edgecolors="black",
                        linewidth=0.3,
                        zorder=10,
                    )

    # Customize plot
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel(f"{metric.upper()} Value")

    # Set x-axis with proper spacing
    horizon_positions = list(range(len(horizons)))
    ax.set_xticks(horizon_positions)
    ax.set_xticklabels([str(h) for h in sorted(horizons)])

    # Set x-axis limits to provide proper spacing around groups
    if len(horizons) > 1:
        ax.set_xlim(-0.5, len(horizons) - 0.5)
    else:
        ax.set_xlim(-0.8, 0.8)

    # Set title
    if title:
        ax.set_title(title, pad=20)
    else:
        ax.set_title(f"{metric.upper()} Performance Across Forecast Horizons", pad=20)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Create dual legend
    from matplotlib.patches import Rectangle

    legend_handles = []
    legend_labels = []

    # Architecture section
    for arch in architectures:
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=colors[arch], alpha=0.8, edgecolor="black"))
        legend_labels.append(arch.upper())

    # Add dummy if present
    if dummy_data:
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor="lightgray", alpha=0.8, edgecolor="black"))
        legend_labels.append("Dummy Model")

    # Variant section (using sample gradients)
    sample_gradients = generate_brightness_gradient("#4682B4", len(variants))
    for idx, variant in enumerate(variants):
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=sample_gradients[idx], alpha=0.8, edgecolor="black"))
        legend_labels.append(variant_mapping.get(variant, variant.capitalize()))

    # Create legend
    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        ncol=min(len(legend_handles), 4),
        frameon=False,
        fancybox=True,
        shadow=False,
        columnspacing=1.5,
        handletextpad=0.8,
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


def plot_performance_vs_static_attributes(
    results: dict[str, Any],
    static_df: pd.DataFrame,
    static_attributes: list[str],
    model_names: list[str],
    horizons: list[int],
    metric: str = "NSE",
    color: str = "#4682B4",
    alpha: float = 0.6,
    marker_size: float = 30,
    figsize: tuple[int, int] = (15, 12),
    attribute_labels: dict[str, str] | None = None,
    hue_by_model: dict[str, str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a matrix of scatter plots showing model performance vs static catchment attributes.

    Creates a grid where:
    - Columns = static attributes
    - Rows = forecast horizons
    - Each subplot = all models combined

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        static_df: DataFrame with columns ["gauge_id", *static_attributes]
        static_attributes: List of static attribute column names (max 3 recommended)
        model_names: List of model names to include
        horizons: List of forecast horizons to include
        metric: Performance metric to plot (e.g., "NSE", "RMSE")
        color: Color for all scatter points (ignored if hue_by_model is provided)
        alpha: Transparency of scatter points
        marker_size: Size of scatter points
        figsize: Figure size as (width, height)
        attribute_labels: Dictionary mapping static attribute names to custom labels (optional)
        hue_by_model: Dictionary mapping model names to colors for model-based coloring (optional)

    Returns:
        Tuple of (figure, axes_array)
    """

    # Validate inputs
    missing_attrs = [attr for attr in static_attributes if attr not in static_df.columns]
    if missing_attrs:
        raise ValueError(f"Static attributes not found in static_df: {missing_attrs}")

    if len(static_attributes) > 3:
        print("Warning: More than 3 static attributes may result in crowded plots")

    # Filter model names to only include those that exist in results
    available_models = [name for name in model_names if name in results]
    if not available_models:
        raise ValueError(f"None of the specified model names found in results. Available: {list(results.keys())}")

    if len(available_models) < len(model_names):
        missing_models = [name for name in model_names if name not in results]
        print(f"Warning: Skipping missing models: {missing_models}")

    # Pre-compute static attribute lookups for faster access
    static_lookups = {}
    for attr in static_attributes:
        static_lookups[attr] = static_df.set_index("gauge_id")[attr].to_dict()

    # Create figure with subplots
    n_rows = len(horizons)
    n_cols = len(static_attributes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Collect all data points for each subplot
    subplot_data = {}
    for row_idx, horizon in enumerate(horizons):
        for col_idx, static_attr in enumerate(static_attributes):
            subplot_data[(row_idx, col_idx)] = {"x": [], "y": [], "models": []}

    # Process each model and collect data points
    for model_name in available_models:
        model_results = results[model_name]

        if "metrics_by_gauge" in model_results:
            # FAST PATH: Use pre-computed metrics
            metrics_by_gauge = model_results["metrics_by_gauge"]

            for gauge_id, gauge_data in metrics_by_gauge.items():
                # Check if gauge exists in all static lookups
                if not all(gauge_id in static_lookups[attr] for attr in static_attributes):
                    continue

                for row_idx, horizon in enumerate(horizons):
                    if horizon in gauge_data and metric.lower() in gauge_data[horizon]:
                        metric_val = gauge_data[horizon][metric.lower()]

                        # Skip NaN values
                        if not np.isnan(metric_val):
                            # Add data point to each column (static attribute)
                            for col_idx, static_attr in enumerate(static_attributes):
                                static_val = static_lookups[static_attr][gauge_id]
                                if not np.isnan(static_val):
                                    subplot_data[(row_idx, col_idx)]["x"].append(static_val)
                                    subplot_data[(row_idx, col_idx)]["y"].append(metric_val)
                                    subplot_data[(row_idx, col_idx)]["models"].append(model_name)

        else:
            # SLOW PATH: Calculate metrics on the fly (fallback)
            if "predictions_df" not in model_results:
                continue

            predictions_df = model_results["predictions_df"]

            # Filter for specified horizons
            filtered_df = predictions_df[predictions_df["horizon"].isin(horizons)]

            if filtered_df.empty:
                continue

            # Calculate metrics efficiently
            for row_idx, horizon in enumerate(horizons):
                horizon_df = filtered_df[filtered_df["horizon"] == horizon]

                for gauge_id, gauge_df in horizon_df.groupby("gauge_id"):
                    # Check if gauge exists in all static lookups
                    if not all(gauge_id in static_lookups[attr] for attr in static_attributes):
                        continue

                    if not gauge_df.empty:
                        observed = gauge_df["observed"].values
                        predicted = gauge_df["predicted"].values

                        # Calculate metric
                        if metric.lower() == "nse":
                            from .metrics import calculate_nse

                            metric_val = calculate_nse(predicted, observed)
                        elif metric.lower() == "rmse":
                            from .metrics import calculate_rmse

                            metric_val = calculate_rmse(predicted, observed)
                        elif metric.lower() == "mae":
                            from .metrics import calculate_mae

                            metric_val = calculate_mae(predicted, observed)
                        else:
                            # Default to NSE
                            from .metrics import calculate_nse

                            metric_val = calculate_nse(predicted, observed)

                        if not np.isnan(metric_val):
                            # Add data point to each column (static attribute)
                            for col_idx, static_attr in enumerate(static_attributes):
                                static_val = static_lookups[static_attr][gauge_id]
                                if not np.isnan(static_val):
                                    subplot_data[(row_idx, col_idx)]["x"].append(static_val)
                                    subplot_data[(row_idx, col_idx)]["y"].append(metric_val)
                                    subplot_data[(row_idx, col_idx)]["models"].append(model_name)

    # Create scatter plots
    for row_idx, horizon in enumerate(horizons):
        for col_idx, static_attr in enumerate(static_attributes):
            ax = axes[row_idx, col_idx]

            # Get data for this subplot
            x_data = subplot_data[(row_idx, col_idx)]["x"]
            y_data = subplot_data[(row_idx, col_idx)]["y"]
            model_data = subplot_data[(row_idx, col_idx)]["models"]

            if x_data and y_data:
                if hue_by_model is not None:
                    # Color by model
                    colors_for_points = [hue_by_model.get(model, color) for model in model_data]
                    ax.scatter(
                        x_data,
                        y_data,
                        c=colors_for_points,
                        s=marker_size,
                        alpha=alpha,
                        edgecolors="black",
                        linewidth=0.3,
                    )
                else:
                    # Use single color
                    ax.scatter(
                        x_data,
                        y_data,
                        c=color,
                        s=marker_size,
                        alpha=alpha,
                        edgecolors="black",
                        linewidth=0.3,
                    )
            else:
                # No data message
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray"
                )

            # Add grid
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

            # Set labels
            # Get custom label or use formatted attribute name
            if attribute_labels and static_attr in attribute_labels:
                attr_label = attribute_labels[static_attr]
            else:
                attr_label = static_attr.replace("_", " ").title()

            # Column headers (static attributes) only on top row
            if row_idx == 0:
                ax.set_title(attr_label, pad=10, fontsize=12)

            # Row labels (horizons) only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{horizon}-day\n{metric.upper()}", fontsize=11)

            # X-axis labels only on bottom row
            if row_idx == len(horizons) - 1:
                ax.set_xlabel(attr_label, fontsize=11)

    # Add legend for models if hue_by_model is provided
    if hue_by_model is not None:
        legend_handles = []
        legend_labels = []

        for model_name in available_models:
            if model_name in hue_by_model:
                legend_handles.append(
                    plt.scatter(
                        [],
                        [],
                        c=hue_by_model[model_name],
                        s=marker_size,
                        alpha=alpha,
                        edgecolors="black",
                        linewidth=0.3,
                    )
                )
                legend_labels.append(model_name)

        if legend_handles:
            fig.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=min(len(legend_handles), 2),
                frameon=False,
            )

    # Adjust layout
    plt.tight_layout()

    return fig, axes
