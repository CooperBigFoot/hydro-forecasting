import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from data.caravanify_parquet import (
    CaravanifyParquet,
    CaravanifyParquetConfig,
)


def main():
    # Define regions to process
    regions = [
        "CL",
        "CA",
        "USA",
        "camelsaus",
        "camelsgb",
        "camelsbr",
        "hysets",
        "lamah",
    ]

    # Anthropogenic attributes to extract
    anthro_columns = [
        "ppd_pk_sav",  # Population density
        "urb_pc_sse",  # Urban extent
        "rdd_mk_sav",  # Road density
        "nli_ix_sav",  # Nighttime lights
        "gdp_ud_sav",  # GDP
        "hdi_ix_sav",  # Human development index
        "dor_pc_pva",  # Degree of regulation
        "rev_mc_usu",  # Reservoir volume
    ]

    # Dictionary to store data from each region
    all_region_data = {}

    # Load data for each region
    for region in regions:
        print(f"Loading data for {region}...")

        # Configure Caravanify
        config = CaravanifyParquetConfig(
            attributes_dir=f"/Users/cooper/Desktop/CaravanifyParquet/{region}/post_processed/attributes",
            timeseries_dir=f"/Users/cooper/Desktop/CaravanifyParquet/{region}/post_processed/timeseries/csv",
            gauge_id_prefix=region,
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        try:
            caravan = CaravanifyParquet(config)
            ids = caravan.get_all_gauge_ids()
            print(f"  Found {len(ids)} stations for {region}")

            if not ids:
                print(f"  No data found for {region}, skipping")
                continue

            caravan.load_stations(ids)
            static_data = caravan.get_static_attributes()

            # Store in dictionary
            all_region_data[region] = static_data
        except Exception as e:
            print(f"  Error loading data for {region}: {e}")

    # Combine data from all regions
    if not all_region_data:
        raise ValueError("No data loaded from any region")

    all_static_data = pd.concat(all_region_data.values(), ignore_index=True)
    print(
        f"Combined data from {len(all_region_data)} regions: {len(all_static_data)} catchments"
    )

    # Check which anthropogenic attributes are available
    available_anthro = [col for col in anthro_columns if col in all_static_data.columns]
    print(f"Available anthropogenic attributes: {available_anthro}")

    # Create subset with only anthropogenic attributes and gauge_id
    anthro_data = all_static_data[["gauge_id"] + available_anthro].copy()

    # Check for missing values
    missing_values = anthro_data.isna().sum()
    print("\nMissing values per attribute:")
    print(missing_values)

    # Impute missing values with median
    for col in available_anthro:
        if anthro_data[col].isna().any():
            median_val = anthro_data[col].median()
            anthro_data[col].fillna(median_val, inplace=True)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    norm_data = pd.DataFrame(
        scaler.fit_transform(anthro_data[available_anthro]), columns=available_anthro
    )

    # Add gauge_id back to normalized data
    norm_data["gauge_id"] = anthro_data["gauge_id"].values

    # Define raw weights for the anthropogenic attributes
    weights_raw = {
        "ppd_pk_sav": 1,
        "urb_pc_sse": 1,
        "rdd_mk_sav": 1,
        "nli_ix_sav": 1,
        "gdp_ud_sav": 3,
        "hdi_ix_sav": 1,
        "dor_pc_pva": 5,
        "rev_mc_usu": 5,
    }

    # Normalize weights
    weights = {attr: 1 / len(available_anthro) for attr in available_anthro}
    for attr in available_anthro:
        if attr in weights_raw:
            weights[attr] = weights_raw[attr] / sum(weights_raw.values())
        else:
            weights[attr] = 0

    assert sum(weights.values()) == 1, "Weights should sum up to 1"

    print("\nWeights assigned to each attribute:")
    print(weights)

    # Calculate HII (weighted sum of normalized attributes)
    hii_values = np.zeros(len(norm_data))
    for attr in available_anthro:
        hii_values += weights[attr] * norm_data[attr].values

    # Add HII to the normalized data and scale to [0,1]
    norm_data["hii"] = hii_values / np.max(hii_values)

    # Define thresholds for categories (using tertiles)
    low_threshold = np.percentile(norm_data["hii"], 30)
    high_threshold = np.percentile(norm_data["hii"], 75)

    # Assign categories
    def assign_category(hii):
        if hii < low_threshold:
            return "Low"
        elif hii < high_threshold:
            return "Medium"
        else:
            return "High"

    norm_data["human_influence_category"] = norm_data["hii"].apply(assign_category)

    # Count catchments in each category
    category_counts = norm_data["human_influence_category"].value_counts()
    print("\nCount of catchments in each category:")
    print(category_counts)

    # Create result DataFrame with only the required columns
    result_df = norm_data[["gauge_id", "hii", "human_influence_category"]]

    # Save to CSV
    output_dir = Path(
        "/Users/cooper/Desktop/hydro-forecasting/scripts/human_influence_index/results"
    )
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "human_influence_classification.parquet"
    result_df.to_parquet(output_path, index=False)
    print(f"\nClassification results saved to {output_path}")

    # Compute bins with Sturges' formula
    n = len(result_df["gauge_id"].unique())
    print(f"Number of unique gauge IDs: {n}")

    bins = int(np.ceil(np.log2(n) + 1))

    # Create histogram of HII values
    sns.set_context("paper", font_scale=1.3)
    plt.figure(figsize=(10, 6))
    plt.hist(norm_data["hii"], bins=30, color="skyblue", edgecolor="black")
    plt.axvline(
        low_threshold,
        color="green",
        linestyle="--",
        label=f"Low threshold: {low_threshold:.3f}",
    )
    plt.axvline(
        high_threshold,
        color="red",
        linestyle="--",
        label=f"High threshold: {high_threshold:.3f}",
    )
    plt.xlabel("Human Influence Index (HII)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Human Influence Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()

    # Save plot
    plot_path = output_dir / "human_influence_histogram.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Histogram saved to {plot_path}")

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
