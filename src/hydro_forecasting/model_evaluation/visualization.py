# Replace contextily with cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable


def plot_rolling_forecast(
    df: pd.DataFrame,
    horizon: int,
    group_identifier: str,
    fig_size: tuple = (12, 6),
    title: str = "",
    color_scheme: dict[str, str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a rolling forecast plot for a specific basin and horizon.

    Args:
        df: DataFrame with prediction results (has columns: 'horizon', 'date', 'prediction', 'observed', 'basin_id')
        horizon: Forecast horizon to visualize
        group_identifier: Basin ID or group identifier to plot
        fig_size: Figure size as (width, height)
        title: Custom title for the plot
        color_scheme: Optional dictionary with colors for 'observed' and 'prediction'

    Returns:
        Tuple containing the figure and axes objects
    """
    # Set default color scheme if not provided
    if color_scheme is None:
        color_scheme = {"observed": "blue", "prediction": "red"}

    # Filter for the specific basin and horizon
    basin_df = df[(df["basin_id"] == group_identifier) & (df["horizon"] == horizon)]

    if basin_df.empty:
        available_ids = df["basin_id"].unique()
        raise ValueError(f"Group identifier '{group_identifier}' not found in results. Available IDs: {available_ids}")

    # Create plot with Seaborn style
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot observations and predictions
    ax.plot(
        basin_df["date"],
        basin_df["observed"],
        color=color_scheme["observed"],
        label="Observed",
        linewidth=2,
    )

    ax.plot(
        basin_df["date"],
        basin_df["prediction"],
        color=color_scheme["prediction"],
        alpha=0.8,
        label=f"{horizon}-Day Forecast",
        linewidth=2,
    )

    # Set title and labels
    if title is None:
        title = f"{horizon}-day Forecast for {group_identifier}"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Streamflow [mm/d]", fontsize=12)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Add legend and formatting
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.7)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def plot_metric_boxplot(
    evaluator_results: dict[str, dict],
    model_names: list[str],
    metric: str = "NSE",
    horizons: list[int] | None = None,
    fig_size: tuple[int, int] = (14, 7),
    title: str | None = None,
    palette: str | list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a boxplot comparing multiple models' metrics across horizons.

    Args:
        evaluator_results: Dictionary with evaluator results for multiple models
        model_names: List of model names to include in the plot
        metric: Metric to visualize (default: "NSE")
        horizons: List of horizons to include (if None, use all available horizons)
        fig_size: Figure size as (width, height)
        title: Custom title for the plot
        palette: Color palette for the plot (string or list of colors)

    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate inputs
    if not model_names:
        raise ValueError("At least one model name must be provided")

    missing_models = [model for model in model_names if model not in evaluator_results]
    if missing_models:
        raise ValueError(f"Models not found in evaluator results: {missing_models}")

    # Create a DataFrame for all models' data
    all_data = []

    for model_name in model_names:
        # Extract basin metrics for this model
        basin_metrics = evaluator_results[model_name]["basin_metrics"]

        # Flatten the nested dictionary to a DataFrame
        for basin, horizon_data in basin_metrics.items():
            for horizon, metrics_dict in horizon_data.items():
                if metric in metrics_dict:
                    all_data.append(
                        {
                            "model": model_name,
                            "basin_id": basin,
                            "horizon": horizon,
                            "metric_value": metrics_dict[metric],
                        }
                    )

    df = pd.DataFrame(all_data)

    # Filter by horizons if specified
    if horizons:
        df = df[df["horizon"].isin(horizons)]

    # Determine available horizons after filtering
    available_horizons = sorted(df["horizon"].unique())

    # Set up color palette
    if palette is None:
        # Default color palette based on number of models
        if len(model_names) == 1:
            palette = "Blues"
        else:
            # Use different color palettes for multiple models
            palette = sns.color_palette("husl", len(model_names))

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Create boxplot
    sns.boxplot(x="horizon", y="metric_value", hue="model", data=df, palette=palette, ax=ax)

    # Add median value labels for each model and horizon
    if len(model_names) <= 3:  # Only add labels if not too many models
        for h_idx, horizon in enumerate(available_horizons):
            for m_idx, model in enumerate(model_names):
                # Filter data for this horizon and model
                filtered = df[(df["horizon"] == horizon) & (df["model"] == model)]
                if not filtered.empty:
                    median = filtered["metric_value"].median()

                    # Calculate position for the text
                    model_count = len(model_names)
                    box_width = 0.8  # Default box width in seaborn
                    position = h_idx

                    # Calculate offset based on model index
                    if model_count > 1:
                        step = box_width / model_count
                        offset = step * (m_idx - (model_count - 1) / 2)
                        position += offset

                    ax.text(
                        position,
                        median + 0.02,
                        f"{median:.2f}",
                        ha="center",
                        va="bottom",
                        color="black",
                        fontweight="bold",
                        fontsize=8,
                    )

    # Set title and labels
    ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
    ax.set_ylabel(f"{metric} Value", fontsize=12)

    if title is None:
        if len(model_names) == 1:
            title = f"Distribution of {metric} Values by Horizon for {model_names[0]}"
        else:
            title = f"Distribution of {metric} Values by Horizon"
    ax.set_title(title, fontsize=14)

    # Customize legend - place at bottom with 4 columns
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True,
        fontsize=10,
    )

    # Add grid lines for better readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Apply styling
    sns.despine()
    plt.tight_layout()

    return fig, ax


def plot_metric_cdf(
    evaluator_results: dict[str, dict],
    model_names: list[str],
    metric: str = "NSE",
    horizon: int = 1,
    fig_size: tuple[int, int] = (10, 6),
    title: str | None = None,
    colors: list[str] | None = None,
    threshold_lines: list[float] | None = None,
    threshold_labels: list[str] | None = None,
    include_median_lines: bool = True,
    legend_loc: str = "best",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the cumulative distribution functions (CDFs) of a metric for multiple models.

    Args:
        evaluator_results: Dictionary with evaluator results for multiple models
        model_names: List of model names to include in the plot
        metric: Metric to visualize (default: "NSE")
        horizon: Forecast horizon to visualize
        fig_size: Figure size as (width, height)
        title: Custom title for the plot
        colors: Optional list of colors for each model's CDF
        threshold_lines: Optional list of thresholds to mark on the plot
        threshold_labels: Optional labels for threshold lines
        include_median_lines: Whether to show vertical lines at median values
        legend_loc: Location for the legend

    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate inputs
    if not model_names:
        raise ValueError("At least one model name must be provided")

    missing_models = [model for model in model_names if model not in evaluator_results]
    if missing_models:
        raise ValueError(f"Models not found in evaluator results: {missing_models}")

    # Set up colors if not provided
    if colors is None:
        # Use a colorblind-friendly palette
        colors = sns.color_palette("colorblind", len(model_names))
    elif len(colors) < len(model_names):
        # If not enough colors provided, cycle through the provided ones
        colors = colors * (len(model_names) // len(colors) + 1)
        colors = colors[: len(model_names)]

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Flag to track if a median line has been added to the legend
    median_in_legend = False

    # Plot CDF for each model
    model_medians = {}
    for i, model_name in enumerate(model_names):
        # Extract basin metrics for this model
        basin_metrics = evaluator_results[model_name]["basin_metrics"]

        # Extract metric values for the specified horizon
        metric_values = []
        for basin, horizon_data in basin_metrics.items():
            if horizon in horizon_data and metric in horizon_data[horizon]:
                value = horizon_data[horizon][metric]
                # Skip NaN values
                if not np.isnan(value):
                    metric_values.append(value)

        if not metric_values:
            print(f"Warning: No data available for model '{model_name}', metric '{metric}' at horizon {horizon}")
            continue

        # Sort values for CDF
        sorted_values = np.sort(metric_values)

        # Calculate cumulative probabilities
        n = len(sorted_values)
        cumulative_prob = np.arange(1, n + 1) / n

        # Store median for later use
        median_value = np.median(sorted_values)
        model_medians[model_name] = median_value

        # Plot CDF
        ax.plot(
            sorted_values,
            cumulative_prob,
            color=colors[i],
            linewidth=2.5,
            label=f"{model_name} (n={n})",
        )

        # Add median line if requested
        if include_median_lines:
            if not median_in_legend:
                # Add the first median line to the legend
                ax.axvline(
                    x=median_value,
                    color="black",
                    linestyle="--",
                    alpha=0.7,
                    label="Median",
                )
                median_in_legend = True
            else:
                # Don't add subsequent median lines to the legend
                ax.axvline(x=median_value, color="black", linestyle="--", alpha=0.7)

    # Add threshold lines if provided
    if threshold_lines:
        if not threshold_labels:
            threshold_labels = [f"Threshold: {t}" for t in threshold_lines]

        for threshold, label in zip(threshold_lines, threshold_labels, strict=False):
            # Add vertical line at threshold
            ax.axvline(x=threshold, color="gray", linestyle="-.", alpha=0.7, label=label)

    # Set labels and title
    ax.set_xlabel(f"{metric} Value", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)

    if title is None:
        title = f"CDF of {metric} for {horizon}-day Forecast"
    ax.set_title(title, fontsize=14)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle="--")

    # Only add legend if there's something to show
    if len(model_names) > 0:
        ax.legend(loc=legend_loc)

    # Apply styling
    sns.despine()
    plt.tight_layout()

    return fig, ax


def plot_basin_map(
    evaluator_results: dict[str, dict],
    model_name: str,
    metric: str = "NSE",
    horizon: int = 1,
    gauge_ids: list[str] | None = None,
    caravanify_instance=None,
    fig_size: tuple[int, int] = (12, 10),
    cmap: str = "RdYlBu",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    basemap: bool = True,
    legend_title: str | None = None,
    threshold: float | None = None,
    show_axes: bool = True,
    grid_alpha: float = 0.5,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot basins on a map colored by their performance metric.

    Args:
        evaluator_results: Dictionary with evaluator results
        model_name: Name of the model to visualize
        metric: Metric to use for coloring (default: "NSE")
        horizon: Forecast horizon to visualize
        gauge_ids: Optional list of gauge IDs to plot (if None, plot all)
        caravanify_instance: Instance of Caravanify to get shapefiles
        fig_size: Figure size as (width, height)
        cmap: Colormap for basin coloring
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        title: Custom title for the plot
        basemap: Whether to add a basemap
        legend_title: Title for the colorbar legend
        threshold: Optional threshold value to highlight
        show_axes: Whether to display latitude and longitude axes
        grid_alpha: Transparency of the coordinate grid

    Returns:
        Tuple containing the figure and axes objects
    """
    if not caravanify_instance:
        raise ValueError("A Caravanify instance is required to access basin shapefiles")

    # Get basin metrics for the specified model
    if model_name not in evaluator_results:
        raise ValueError(f"Model '{model_name}' not found in evaluator results")

    basin_metrics = evaluator_results[model_name]["basin_metrics"]

    # Extract metrics for the specified horizon
    metric_values = {}
    for basin, horizon_data in basin_metrics.items():
        if horizon in horizon_data and metric in horizon_data[horizon]:
            value = horizon_data[horizon][metric]
            if not np.isnan(value):
                metric_values[basin] = value

    if not metric_values:
        raise ValueError(f"No data available for metric '{metric}' at horizon {horizon}")

    # Filter to specific gauge IDs if provided
    if gauge_ids:
        metric_values = {basin: value for basin, value in metric_values.items() if basin in gauge_ids}
        if not metric_values:
            raise ValueError(f"None of the specified gauge IDs have data for metric '{metric}' at horizon {horizon}")

    # Get basin shapefiles
    try:
        all_shapefiles = caravanify_instance.get_shapefiles()
        basin_gdf = all_shapefiles[all_shapefiles["gauge_id"].isin(metric_values.keys())].copy()
        if basin_gdf.empty:
            raise ValueError("No basin shapefiles found for the specified gauge IDs")
        if basin_gdf.crs is not None and basin_gdf.crs != "EPSG:4326":
            basin_gdf = basin_gdf.to_crs("EPSG:4326")
    except Exception as e:
        raise ValueError(f"Error getting basin shapefiles: {e}")

    basin_gdf["metric_value"] = basin_gdf["gauge_id"].map(metric_values)
    x_min, y_min, x_max, y_max = basin_gdf.total_bounds
    padding = 0.1
    x_range, y_range = x_max - x_min, y_max - y_min
    x_min, x_max, y_min, y_max = (
        x_min - padding * x_range,
        x_max + padding * x_range,
        y_min - padding * y_range,
        y_max + padding * y_range,
    )

    # Create figure with cartopy projection
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": projection})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    # Plot basins
    norm = mcolors.Normalize(
        vmin=vmin or basin_gdf["metric_value"].min(),
        vmax=vmax or basin_gdf["metric_value"].max(),
    )
    basin_gdf.plot(
        column="metric_value",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )

    # Set map extent
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(cmap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(legend_title or f"{metric} Value", size=12)

    # Set title
    ax.set_title(title or f"{metric} at {horizon}-day Horizon for {model_name}", fontsize=14)

    # Configure grid and axes
    if show_axes:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=grid_alpha, linestyle="--")
        gl.top_labels, gl.right_labels = False, False
        gl.xlabel_style, gl.ylabel_style = {"size": 8}, {"size": 8}
    else:
        ax.set_axis_off()

    return fig, ax


def plot_basin_difference_map(
    evaluator_results: dict[str, dict],
    model1_name: str,
    model2_name: str,
    metric: str = "NSE",
    horizon: int = 1,
    gauge_ids: list[str] | None = None,
    caravanify_instance=None,
    fig_size: tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",  # Red-Blue reversed (red=negative, blue=positive differences)
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    basemap: bool = True,
    difference_legend_title: str | None = None,
    threshold: float | None = None,
    show_axes: bool = True,
    grid_alpha: float = 0.5,
    hist_in_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot basins on a map colored by the difference in performance metrics between two models.

    Args:
        evaluator_results: Dictionary with evaluator results
        model1_name: Name of the first model (baseline)
        model2_name: Name of the second model to compare against baseline
        metric: Metric to use for comparison (default: "NSE")
        horizon: Forecast horizon to visualize
        gauge_ids: Optional list of gauge IDs to plot (if None, plot all)
        caravanify_instance: Instance of Caravanify to get shapefiles
        fig_size: Figure size as (width, height)
        cmap: Colormap for difference coloring (default: "RdBu_r")
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        title: Custom title for the plot
        basemap: Whether to add a basemap
        difference_legend_title: Title for the colorbar legend
        threshold: Optional threshold value to highlight significant differences
        show_axes: Whether to display latitude and longitude axes
        grid_alpha: Transparency of the coordinate grid
        hist_in_legend: Whether to include a histogram in the legend

    Returns:
        Tuple containing the figure and axes objects
    """
    if not caravanify_instance:
        raise ValueError("A Caravanify instance is required to access basin shapefiles")

    # Validate inputs
    for model_name in [model1_name, model2_name]:
        if model_name not in evaluator_results:
            raise ValueError(f"Model '{model_name}' not found in evaluator results")

    # Get basin metrics for both models
    basin_metrics1 = evaluator_results[model1_name]["basin_metrics"]
    basin_metrics2 = evaluator_results[model2_name]["basin_metrics"]

    # Extract metric values for the specified horizon
    metric_values1 = {}
    metric_values2 = {}
    for basin, horizon_data in basin_metrics1.items():
        if horizon in horizon_data and metric in horizon_data[horizon]:
            value = horizon_data[horizon][metric]
            if not np.isnan(value):
                metric_values1[basin] = value

    for basin, horizon_data in basin_metrics2.items():
        if horizon in horizon_data and metric in horizon_data[horizon]:
            value = horizon_data[horizon][metric]
            if not np.isnan(value):
                metric_values2[basin] = value

    # Ensure there are common basins to compare
    common_basins = set(metric_values1.keys()) & set(metric_values2.keys())
    if not common_basins:
        raise ValueError(f"No common basins found between models for metric '{metric}' at horizon {horizon}")

    # Calculate metric differences (model2 - model1)
    metric_differences = {}
    for basin in common_basins:
        if gauge_ids and basin not in gauge_ids:
            continue
        metric_differences[basin] = metric_values2[basin] - metric_values1[basin]

    if not metric_differences:
        raise ValueError("No basins to plot after filtering")

    # Get basin shapefiles
    try:
        all_shapefiles = caravanify_instance.get_shapefiles()
        basin_gdf = all_shapefiles[all_shapefiles["gauge_id"].isin(metric_differences.keys())].copy()
        if basin_gdf.empty:
            raise ValueError("No basin shapefiles found for the specified gauge IDs")
        if basin_gdf.crs is not None and basin_gdf.crs != "EPSG:4326":
            basin_gdf = basin_gdf.to_crs("EPSG:4326")
    except Exception as e:
        raise ValueError(f"Error getting basin shapefiles: {e}")

    # Add metric differences to GeoDataFrame
    basin_gdf["metric_diff"] = basin_gdf["gauge_id"].map(metric_differences)

    # Calculate map bounds
    x_min, y_min, x_max, y_max = basin_gdf.total_bounds
    padding = 0.1
    x_range, y_range = x_max - x_min, y_max - y_min
    x_min, x_max, y_min, y_max = (
        x_min - padding * x_range,
        x_max + padding * x_range,
        y_min - padding * y_range,
        y_max + padding * y_range,
    )

    # Set colormap range to be symmetric around zero if not specified
    if vmin is None and vmax is None:
        abs_max = max(abs(min(metric_differences.values())), abs(max(metric_differences.values())))
        vmin, vmax = -abs_max, abs_max
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    # Create figure with cartopy projection
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": projection})

    # Add basemap features
    if basemap:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    # Define colormap with special handling for threshold
    if threshold is not None and threshold > 0:
        # Create a custom colormap with white/neutral color in the insignificant range
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, 256))
        # Find the midpoint in the colormap (representing zero difference)
        mid_idx = int(256 * (-vmin) / (vmax - vmin))

        # Create transition zones around zero based on threshold
        neg_threshold_idx = int(256 * (-vmin - threshold) / (vmax - vmin))
        pos_threshold_idx = int(256 * (-vmin + threshold) / (vmax - vmin))

        # Set colors in the insignificant range to a light gray
        for i in range(neg_threshold_idx, pos_threshold_idx + 1):
            if 0 <= i < 256:
                colors[i] = mcolors.to_rgba("lightgray")

        # Create a new colormap
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_diff", colors)
        cmap_to_use = custom_cmap
    else:
        cmap_to_use = cmap

    # Create the norm for the colormap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot basins with difference colors
    basin_gdf.plot(
        column="metric_diff",
        ax=ax,
        cmap=cmap_to_use,
        norm=norm,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )

    # Set map extent
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())

    # Add histogram in colorbar if requested
    if hist_in_legend:
        # Create the main color bar
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(cmap_to_use))
        sm.set_array([])

        # Create a separate axes for the colorbar
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)

        # Set colorbar label
        cbar.set_label(
            difference_legend_title or f"Δ{metric} ({model2_name} - {model1_name})",
            size=12,
        )

        # Add a small histogram to show the distribution of differences
        hist_ax = fig.add_axes([0.92, 0.1, 0.06, 0.1])  # Position below colorbar
        hist_values = list(metric_differences.values())
        hist_ax.hist(hist_values, bins=10, color="gray", alpha=0.7)
        hist_ax.set_title("Distribution", fontsize=8)
        hist_ax.tick_params(axis="both", which="major", labelsize=6)
        hist_ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

        # Add threshold lines if provided
        if threshold is not None:
            hist_ax.axvline(x=threshold, color="blue", linestyle=":", linewidth=0.8)
            hist_ax.axvline(x=-threshold, color="red", linestyle=":", linewidth=0.8)
    else:
        # Standard colorbar without histogram
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(cmap_to_use))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(
            difference_legend_title or f"Δ{metric} ({model2_name} - {model1_name})",
            size=12,
        )

    # Add summary statistics to plot
    pos_count = sum(1 for x in metric_differences.values() if x > 0)
    neg_count = sum(1 for x in metric_differences.values() if x < 0)
    if threshold is not None:
        sig_pos_count = sum(1 for x in metric_differences.values() if x > threshold)
        sig_neg_count = sum(1 for x in metric_differences.values() if x < -threshold)
        insig_count = sum(1 for x in metric_differences.values() if -threshold <= x <= threshold)

        stats_text = (
            f"Model Comparison Summary:\n"
            f"{model2_name} better: {pos_count} basins ({sig_pos_count} significant)\n"
            f"{model1_name} better: {neg_count} basins ({sig_neg_count} significant)\n"
            f"Insignificant difference: {insig_count} basins"
        )
    else:
        stats_text = (
            f"Model Comparison Summary:\n"
            f"{model2_name} better: {pos_count} basins\n"
            f"{model1_name} better: {neg_count} basins"
        )

    # Add the stats text box
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        fontsize=9,
        verticalalignment="bottom",
    )

    # Set title
    ax.set_title(
        title or f"Difference in {metric} at {horizon}-day Horizon: {model2_name} vs {model1_name}",
        fontsize=14,
    )

    # Configure grid and axes
    if show_axes:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=grid_alpha, linestyle="--")
        gl.top_labels, gl.right_labels = False, False
        gl.xlabel_style, gl.ylabel_style = {"size": 8}, {"size": 8}
    else:
        ax.set_axis_off()

    return fig, ax
