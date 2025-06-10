from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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
                basin_metrics = model_data[arch][variant]["basin_metrics"]

                # Extract metric values for this horizon across all basins
                values = []
                for basin_data in basin_metrics.values():
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
    bar_width = 0.15
    group_width = n_variants * bar_width
    x_positions = np.arange(n_architectures)

    # Plot bars
    for variant_idx, variant in enumerate(variants):
        x_offset = (variant_idx - (n_variants - 1) / 2) * bar_width

        medians = []
        stds = []
        colors_list = []

        for arch_idx, arch in enumerate(architectures):
            if arch in performance_data and variant in performance_data[arch]:
                medians.append(performance_data[arch][variant]["median"])
                stds.append(performance_data[arch][variant]["std"])
                colors_list.append(colors[arch])
            else:
                medians.append(0)
                stds.append(0)
                colors_list.append("#CCCCCC")  # Gray for missing data

        # Create bars
        bars = ax.bar(
            x_positions + x_offset,
            medians,
            bar_width,
            yerr=stds,
            capsize=3,
            color=colors_list,
            hatch=patterns.get(variant),
            edgecolor="black",
            linewidth=0.8,
            label=variant.capitalize(),
            alpha=0.8 if patterns.get(variant) else 1.0,
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
        edgecolor="gray"
    )

    # Adjust layout
    plt.tight_layout()

    return fig, ax
