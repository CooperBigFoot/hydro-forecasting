#!/usr/bin/env python3
"""
Custom cluster visualization script with configurable plotting parameters.
Loads cluster assignments and reprocesses time series data with caching support.
"""

import gc
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

matplotlib.use("Agg")

from hydro_forecasting.clustering.preprocess_time_series import prepare_timeseries_data
from hydro_forecasting.data.caravanify_parquet import (
    CaravanifyParquet,
    CaravanifyParquetConfig,
)


@dataclass
class PlotConfig:
    """Configuration for customizing the cluster plot appearance."""

    # Figure settings
    figure_size: tuple[float, float] = (10, 12)
    dpi: int = 300

    # Grid layout
    max_cols: int = 3

    # Colors
    member_color: str = "gray"
    member_alpha: float = 0.4
    centroid_colors: list[str] | None = None  # If None, uses seaborn colorblind palette
    centroid_linewidth: float = 3.0

    # Line styles
    member_linestyle: str = "-"
    centroid_linestyle: str = "-"
    zero_line_color: str = "black"
    zero_line_style: str = "--"
    zero_line_width: float = 0.5

    # Font settings
    title_fontsize: int = 12
    axis_label_fontsize: int = 10
    tick_label_fontsize: int = 8
    legend_fontsize: int = 10
    font_family: str = "sans-serif"

    # Axis labels
    x_label: str = "Week"
    y_label: str = "Standardized Flow"

    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_color: str = "gray"
    grid_linestyle: str = "-"

    # Legend
    legend_location: str = "lower center"
    legend_ncol: int = 2
    legend_frameon: bool = False
    legend_bbox_anchor: tuple[float, float] = (0.5, 0)

    # Layout
    subplot_adjust_bottom: float = 0.1
    use_tight_layout: bool = True

    # Plot limits
    y_min: float | None = None
    y_max: float | None = None

    # Max series to plot per cluster
    max_series_per_cluster: int = 200

    # Subplot styling
    remove_spines: bool = True  # Uses seaborn despine

    # Save settings
    save_format: str = "png"
    bbox_inches: str = "tight"


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    # Input paths
    cluster_assignments_path: Path
    countries: list[str]
    attributes_base_dir: Path
    timeseries_base_dir: Path

    # Cache settings
    cache_dir: Path = Path("./cache")
    cache_filename: str = "processed_hydrographs.pkl"
    force_reprocess: bool = False

    # Processing parameters
    batch_size: int = 100
    hemisphere_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize hemisphere map if not provided."""
        if not self.hemisphere_map:
            self.hemisphere_map = {
                "CL": "southern",
                "USA": "northern",
                "CH": "northern",
                "camelsaus": "southern",
                "camelsgb": "northern",
                "camelsbr": "southern",
                "hysets": "northern",
                "lamah": "northern",
            }


def load_cached_data(cache_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Load processed hydrographs from cache if available."""
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"Successfully loaded {len(data['basin_ids'])} basins from cache")
            return data["hydrographs"], data["basin_ids"]
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None


def save_cached_data(cache_path: Path, hydrographs: np.ndarray, basin_ids: list[str]):
    """Save processed hydrographs to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"hydrographs": hydrographs, "basin_ids": basin_ids}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(basin_ids)} basins to cache at {cache_path}")


def process_time_series_data(config: DataConfig) -> tuple[np.ndarray, list[str]]:
    """
    Process time series data from multiple countries in batches.
    Returns standardized weekly hydrographs and corresponding basin IDs.
    """
    print("\n=== Processing time series data ===")
    processed_hydrograph_batches = []
    successfully_processed_basin_ids = []
    total_basins_processed = 0

    # Process one country at a time
    for country in config.countries:
        try:
            print(f"\nProcessing country: {country}")

            # Configure CaravanifyParquet to get gauge IDs
            caravan_config = CaravanifyParquetConfig(
                attributes_dir=f"{config.attributes_base_dir}/{country}/post_processed/attributes",
                timeseries_dir=f"{config.timeseries_base_dir}/{country}/post_processed/timeseries/csv",
                gauge_id_prefix=country,
            )
            caravan = CaravanifyParquet(caravan_config)

            # Get all gauge IDs for this country
            country_gauge_ids = caravan.get_all_gauge_ids()
            print(f"Found {len(country_gauge_ids)} gauge IDs for {country}")

            del caravan
            gc.collect()

            if not country_gauge_ids:
                print(f"Skipping {country} as no gauge IDs were found")
                continue

            # Process in batches
            num_batches = (len(country_gauge_ids) + config.batch_size - 1) // config.batch_size

            for batch_idx in tqdm(range(num_batches), desc=f"Processing {country} in batches"):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, len(country_gauge_ids))
                current_batch_ids = country_gauge_ids[start_idx:end_idx]

                if not current_batch_ids:
                    continue

                try:
                    # Configure for this batch
                    batch_caravan_config = CaravanifyParquetConfig(
                        attributes_dir=config.attributes_base_dir / country / "post_processed/attributes",
                        timeseries_dir=config.timeseries_base_dir / country / "post_processed/timeseries/csv",
                        gauge_id_prefix=country,
                    )
                    batch_caravan = CaravanifyParquet(batch_caravan_config)

                    # Load stations in this batch
                    batch_caravan.load_stations(current_batch_ids)

                    # Get time series data
                    daily_data = batch_caravan.get_time_series()

                    if not daily_data.empty:
                        # Process to weekly hydrographs
                        processed_ts, processed_ids = prepare_timeseries_data(
                            df=daily_data,
                            basin_id_col="gauge_id",
                            date_col="date",
                            flow_col="streamflow",
                            standardize=True,
                            hemisphere_map=config.hemisphere_map,
                        )

                        if processed_ts.size > 0:
                            processed_hydrograph_batches.append(processed_ts)
                            successfully_processed_basin_ids.extend(processed_ids)
                            total_basins_processed += len(processed_ids)

                    del daily_data
                    del batch_caravan
                    gc.collect()

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing country {country}: {e}")
            continue

    print(f"\nTotal basins processed: {total_basins_processed}")

    if not processed_hydrograph_batches:
        raise ValueError("No data was successfully processed")

    # Consolidate all batches
    hydrographs = np.vstack(processed_hydrograph_batches)
    return hydrographs, successfully_processed_basin_ids


def compute_cluster_centroids(hydrographs: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Compute cluster centroids using simple averaging.
    For a more sophisticated approach, could implement DBA (Dynamic Time Warping Barycenter Averaging).
    """
    centroids = np.zeros((n_clusters, hydrographs.shape[1]))

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if np.sum(cluster_mask) > 0:
            # Simple mean for now
            centroids[cluster_id] = np.mean(hydrographs[cluster_mask], axis=0)

    return centroids


def plot_clusters(
    hydrographs: np.ndarray,
    basin_ids: list[str],
    cluster_assignments: pd.DataFrame,
    plot_config: PlotConfig,
    save_path: Path | None = None,
):
    """Create customized cluster plot."""

    # Get cluster labels for our basins
    labels = np.zeros(len(basin_ids), dtype=int)
    for idx, basin_id in enumerate(basin_ids):
        if basin_id in cluster_assignments.set_index("gauge_id")["cluster"]:
            labels[idx] = cluster_assignments.set_index("gauge_id").loc[basin_id, "cluster"]

    n_clusters = len(np.unique(labels))

    # Compute centroids
    centroids = compute_cluster_centroids(hydrographs, labels, n_clusters)

    # Set up colors
    if plot_config.centroid_colors is None:
        centroid_colors = sns.color_palette("colorblind", n_clusters)
    else:
        centroid_colors = plot_config.centroid_colors

    # Calculate grid dimensions
    cols = min(plot_config.max_cols, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=plot_config.figure_size, squeeze=False)

    # Set font properties
    plt.rcParams.update(
        {
            "font.family": plot_config.font_family,
            "font.size": plot_config.axis_label_fontsize,
        }
    )

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # Create legend elements
    legend_handles = []
    legend_labels = []

    # Plot each cluster
    for cluster_id in range(n_clusters):
        row, col = cluster_id // cols, cluster_id % cols
        ax = axes[row, col]

        # Get series in this cluster
        cluster_mask = labels == cluster_id
        cluster_series = hydrographs[cluster_mask]

        # Limit number of series
        if len(cluster_series) > plot_config.max_series_per_cluster:
            # Random selection to avoid bias
            indices = np.random.choice(len(cluster_series), plot_config.max_series_per_cluster, replace=False)
            cluster_series = cluster_series[indices]

        # Plot member series
        for i, series in enumerate(cluster_series):
            line = ax.plot(
                series,
                color=plot_config.member_color,
                alpha=plot_config.member_alpha,
                linestyle=plot_config.member_linestyle,
                linewidth=0.5,
            )[0]

            # Add to legend only once
            if cluster_id == 0 and i == 0:
                legend_handles.append(line)
                legend_labels.append("Cluster members")

        # Plot centroid
        centroid_line = ax.plot(
            centroids[cluster_id],
            color=centroid_colors[cluster_id % len(centroid_colors)],
            linewidth=plot_config.centroid_linewidth,
            linestyle=plot_config.centroid_linestyle,
        )[0]

        # Add to legend only once
        if cluster_id == 0:
            legend_handles.append(centroid_line)
            legend_labels.append("Cluster centroid")

        # Styling
        ax.set_title(f"Cluster {cluster_id}", fontsize=plot_config.title_fontsize)

        # Add zero line
        ax.axhline(
            0,
            color=plot_config.zero_line_color,
            linestyle=plot_config.zero_line_style,
            linewidth=plot_config.zero_line_width,
        )

        # Grid
        if plot_config.show_grid:
            ax.grid(
                True, alpha=plot_config.grid_alpha, color=plot_config.grid_color, linestyle=plot_config.grid_linestyle
            )

        # Axis labels
        is_bottom_plot = row == rows - 1 or (col == cluster_id % cols and (cluster_id + cols) >= n_clusters)
        if is_bottom_plot:
            ax.set_xlabel(plot_config.x_label, fontsize=plot_config.axis_label_fontsize)

        if col == 0:
            ax.set_ylabel(plot_config.y_label, fontsize=plot_config.axis_label_fontsize)

        # Set axis limits if specified
        if plot_config.y_min is not None or plot_config.y_max is not None:
            ax.set_ylim(plot_config.y_min, plot_config.y_max)

        # Tick label size
        ax.tick_params(axis="both", labelsize=plot_config.tick_label_fontsize)

        # Remove spines if requested
        if plot_config.remove_spines:
            sns.despine(ax=ax)

    # Hide unused subplots
    for j in range(n_clusters, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Add legend
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc=plot_config.legend_location,
        ncol=plot_config.legend_ncol,
        bbox_to_anchor=plot_config.legend_bbox_anchor,
        frameon=plot_config.legend_frameon,
        fontsize=plot_config.legend_fontsize,
    )

    # Layout adjustments
    if plot_config.use_tight_layout:
        plt.tight_layout()

    plt.subplots_adjust(bottom=plot_config.subplot_adjust_bottom)

    # Save if path provided
    if save_path:
        plt.savefig(
            save_path,
            dpi=plot_config.dpi,
            format=plot_config.save_format,
            bbox_inches=plot_config.bbox_inches,
        )
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    """Main function to create custom cluster plot."""

    # Data configuration
    data_config = DataConfig(
        cluster_assignments_path=Path(
            "/workspace/hydro-forecasting/scripts/cluster_basins/clustering_results/cluster_assignments_shifted_refactor.csv"
        ),
        countries=["CH", "CL", "USA", "camelsaus", "camelsgb", "camelsbr", "hysets", "lamah"],
        attributes_base_dir=Path("/workspace/CaravanifyParquet"),
        timeseries_base_dir=Path("/workspace/CaravanifyParquet"),
        cache_dir=Path("/workspace/hydro-forecasting/scripts/cluster_basins/cache"),
        force_reprocess=False,  # Set to True to reprocess data
    )

    # Plot configuration - easily customizable for thesis style
    plot_config = PlotConfig(
        figure_size=(12, 14),
        dpi=300,
        max_cols=3,
        member_color="gray",
        member_alpha=0.3,
        centroid_linewidth=2.5,
        title_fontsize=14,
        # axis_label_fontsize=12,
        # tick_label_fontsize=10,
        # legend_fontsize=11,
        show_grid=True,
        grid_alpha=0.2,
        subplot_adjust_bottom=0.08,
        max_series_per_cluster=400,
    )


    # Load cluster assignments
    print("\nLoading cluster assignments...")
    cluster_assignments = pd.read_csv(data_config.cluster_assignments_path)
    print(f"Found {len(cluster_assignments)} basin assignments")
    print(f"Number of clusters: {cluster_assignments['cluster'].nunique()}")

    # Check cache or process data
    cache_path = data_config.cache_dir / data_config.cache_filename

    if not data_config.force_reprocess:
        cached_data = load_cached_data(cache_path)
        if cached_data:
            hydrographs, basin_ids = cached_data
        else:
            hydrographs, basin_ids = process_time_series_data(data_config)
            save_cached_data(cache_path, hydrographs, basin_ids)
    else:
        hydrographs, basin_ids = process_time_series_data(data_config)
        save_cached_data(cache_path, hydrographs, basin_ids)

    # Create plot
    output_path = Path(
        "/workspace/hydro-forecasting/scripts/cluster_basins/clustering_results/cluster_plot_custom.png"
    )
    plot_clusters(
        hydrographs=hydrographs,
        basin_ids=basin_ids,
        cluster_assignments=cluster_assignments,
        plot_config=plot_config,
        save_path=output_path,
    )

    print("\nPlot generation complete!")
    print(f"Output saved to: {output_path}")
    print("\nTo customize the plot appearance, modify the PlotConfig parameters in main()")


if __name__ == "__main__":
    main()
