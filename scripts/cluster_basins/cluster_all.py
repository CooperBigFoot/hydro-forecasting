import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import gc
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")


from hydro_forecasting.clustering.preprocess_time_series import prepare_timeseries_data
from hydro_forecasting.clustering.time_series_clusterer import TimeSeriesClusterer
from hydro_forecasting.data.caravanify_parquet import (
    CaravanifyParquet,
    CaravanifyParquetConfig,
)


@dataclass
class ClusteringConfig:
    """Configuration for the time series clustering process."""

    # Data sources
    countries: list[str]
    attributes_base_dir: str
    timeseries_base_dir: str

    # Clustering parameters
    min_clusters: int = 10
    max_clusters: int = 18
    max_iter: int = 75
    n_jobs: int = -1
    warping_window: float = 0.2

    # Output paths
    output_dir: str = "./clustering_results"
    elbow_plot_filename: str = "elbow_plot_shifted_refactor.png"
    cluster_plot_filename: str = "cluster_plot_shifted_refactor.png"
    results_csv_filename: str = "cluster_assignments_shifted_refactor.csv"

    # Optimization method
    optimization_method: str = "elbow"  # 'elbow' or 'silhouette'

    # Batch processing parameters
    batch_size: int = 100

    # Hemisphere map for standardizing time series
    hemisphere_map: dict = None


def main(config: ClusteringConfig):
    """
    Main function to cluster time series data from multiple countries using batch processing.

    Args:
        config: Clustering configuration
    """
    # Create output directory if it doesn't exist
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create full paths for output files
    cluster_plot_path = output_dir / config.cluster_plot_filename
    results_csv_path = output_dir / config.results_csv_filename
    elbow_plot_path = output_dir / config.elbow_plot_filename

    # Initialize hemisphere map if not provided
    if config.hemisphere_map is None:
        config.hemisphere_map = {
            "CL": "southern",
            "USA": "northern",
            "CH": "northern",
            # Add other countries as needed
        }

    print("\n=== Phase 1 & 2: Processing data country-by-country in batches ===")
    # Process each country's data in batches to avoid memory issues
    processed_hydrograph_batches = []  # To store NumPy arrays from each batch
    successfully_processed_basin_ids = []  # To store basin IDs that were processed successfully
    total_basins_processed = 0

    # Process one country at a time
    for country in config.countries:
        try:
            print(f"\nProcessing country: {country}")

            # Configure CaravanifyParquet just to get gauge IDs for this country
            caravan_config = CaravanifyParquetConfig(
                attributes_dir=f"{config.attributes_base_dir}/{country}/post_processed/attributes",
                timeseries_dir=f"{config.timeseries_base_dir}/{country}/post_processed/timeseries/csv",
                gauge_id_prefix=country,
            )
            caravan = CaravanifyParquet(caravan_config)

            # Get all gauge IDs for this country
            country_gauge_ids = caravan.get_all_gauge_ids()
            print(f"Found {len(country_gauge_ids)} gauge IDs for {country}")

            # Clean up the caravan instance used just for collecting IDs
            del caravan
            gc.collect()

            # Skip if no gauge IDs found for this country
            if not country_gauge_ids:
                print(f"Skipping {country} as no gauge IDs were found")
                continue

            # Determine number of batches for this country
            batch_size = config.batch_size
            num_batches = (len(country_gauge_ids) + batch_size - 1) // batch_size  # Ceiling division

            # Process this country's data in batches with progress tracking
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {country} in batches"):
                # Get current batch of IDs for this country
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(country_gauge_ids))
                current_batch_ids = country_gauge_ids[start_idx:end_idx]

                if not current_batch_ids:
                    continue

                try:
                    # Configure CaravanifyParquet for this country batch
                    batch_caravan_config = CaravanifyParquetConfig(
                        attributes_dir=config.attributes_base_dir / country / "post_processed/attributes",
                        timeseries_dir=config.timeseries_base_dir / country / "post_processed/timeseries/csv",
                        gauge_id_prefix=country,
                    )
                    batch_caravan = CaravanifyParquet(batch_caravan_config)

                    # Load only the stations in this batch
                    batch_caravan.load_stations(current_batch_ids)

                    # Get time series data for this batch
                    daily_data_for_batch = batch_caravan.get_time_series()

                    # Process this batch into weekly hydrographs if we have data
                    if not daily_data_for_batch.empty:
                        processed_ts_array, processed_ids_for_batch = prepare_timeseries_data(
                            df=daily_data_for_batch,
                            basin_id_col="gauge_id",
                            date_col="date",
                            flow_col="streamflow",
                            standardize=True,
                            hemisphere_map=config.hemisphere_map,
                        )

                        # Store results if we processed any basins successfully
                        if processed_ts_array.size > 0:
                            processed_hydrograph_batches.append(processed_ts_array)
                            successfully_processed_basin_ids.extend(processed_ids_for_batch)
                            total_basins_processed += len(processed_ids_for_batch)

                    # Explicitly clean up to help garbage collection
                    del daily_data_for_batch
                    del batch_caravan
                    gc.collect()

                except Exception as e:
                    print(
                        f"Error processing {country} batch {batch_idx} (first ID: {current_batch_ids[0] if current_batch_ids else 'N/A'}): {e}"
                    )
                    continue

        except Exception as e:
            print(f"Error processing country {country}: {e}")
            continue

    print(f"\nTotal number of basins successfully processed: {total_basins_processed}")

    print("\n=== Phase 3: Consolidating results ===")
    # Consolidate processed batches
    if not processed_hydrograph_batches:
        raise ValueError("No data was successfully processed. Check input data and error messages.")

    ts_data_standardized = np.vstack(processed_hydrograph_batches)
    basin_ids = successfully_processed_basin_ids

    print(f"Successfully processed {len(basin_ids)} basins into a matrix of shape {ts_data_standardized.shape}")

    # Check for NaN values that might affect clustering
    nan_count = np.isnan(ts_data_standardized).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the standardized data")

    print("\n=== Phase 4: Clustering ===")
    # Proceed with clustering as before
    clusterer = TimeSeriesClusterer(
        max_iter=config.max_iter,
        n_jobs=config.n_jobs,
        warping_window=config.warping_window,
    )

    # Optimize clusters
    print(f"Optimizing number of clusters from {config.min_clusters} to {config.max_clusters}...")
    clusterer.optimize_clusters(ts_data_standardized, config.min_clusters, config.max_clusters)

    # Plot and save optimization results
    plt.figure(figsize=(15, 7))
    clusterer.plot_cluster_optimization(save_path=elbow_plot_path)
    plt.close()
    print(f"Saved optimization plot to {elbow_plot_path}")

    # Get recommended number of clusters
    optimal_clusters = clusterer.recommend_clusters(method=config.optimization_method)
    print(f"Recommended number of clusters: {optimal_clusters}")

    # Fit with recommended clusters
    print(f"Fitting clusterer with {optimal_clusters} clusters...")
    clusterer = TimeSeriesClusterer(
        n_clusters=optimal_clusters,
        max_iter=config.max_iter,
        n_jobs=config.n_jobs,
        warping_window=config.warping_window,
    )
    clusterer.fit(ts_data_standardized, basin_ids)

    # Plot and save cluster results
    clusterer.plot_clusters(max_series_per_cluster=200, save_path=cluster_plot_path)
    plt.close()
    print(f"Saved cluster plot to {cluster_plot_path}")

    # Create mapping of gauge_id to cluster
    id_to_cluster = {id_: clusterer.get_label_from_id(id_) for id_ in basin_ids}

    # Save to CSV
    results_df = pd.DataFrame(
        {
            "gauge_id": list(id_to_cluster.keys()),
            "cluster": list(id_to_cluster.values()),
        }
    )
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    print("\n=== Clustering process completed successfully ===")


if __name__ == "__main__":
    # Define configuration
    config = ClusteringConfig(
        countries=["CH", "CL", "USA"],
        attributes_base_dir=Path("/Users/cooper/Desktop/CaravanifyParquet"),
        timeseries_base_dir=Path("/Users/cooper/Desktop/CaravanifyParquet"),
        output_dir=Path(".scripts/cluster_basins/clustering_results"),
        min_clusters=10,
        max_clusters=18,
        max_iter=75,
        n_jobs=-1,
        warping_window=2 / 52,
        optimization_method="elbow",
        batch_size=100,
        hemisphere_map={
            "CL": "southern",
            "USA": "northern",
            "CH": "northern",
        },
    )

    main(config)
