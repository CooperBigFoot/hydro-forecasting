#!/usr/bin/env python3
"""
Test script to check time periods in cached model predictions.
"""

from pathlib import Path

import pandas as pd

# Path to the cached predictions
cache_path = Path("/Users/cooper/Desktop/hydro-forecasting/data_cache/model_evaluation_cache_kyrgyzstan/predictions")

# Models to check
models = ["tsmixer_benchmark", "tft_benchmark", "ealstm_benchmark", "tide_benchmark", "dummy"]

# Gauge to check
gauge_id = "CA_15102"
horizon = 10

print("Checking time periods for different models...")
print(f"Gauge: {gauge_id}, Horizon: {horizon}")
print("-" * 80)

for model in models:
    parquet_file = cache_path / f"{model}.parquet"

    if parquet_file.exists():
        # Load the predictions
        df = pd.read_parquet(parquet_file)

        # Filter for specific gauge and horizon
        filtered_df = df[(df["gauge_id"] == gauge_id) & (df["horizon"] == horizon)]

        if not filtered_df.empty:
            # Get date range
            min_date = filtered_df["date"].min()
            max_date = filtered_df["date"].max()
            num_samples = len(filtered_df)

            print(f"\n{model}:")
            print(f"  Date range: {min_date} to {max_date}")
            print(f"  Number of samples: {num_samples}")

            # Check for any NaT (Not a Time) values
            nat_count = filtered_df["date"].isna().sum()
            if nat_count > 0:
                print(f"  WARNING: {nat_count} NaT values found!")

            # Show first few and last few dates
            print(f"  First 5 dates: {filtered_df['date'].head().tolist()}")
            print(f"  Last 5 dates: {filtered_df['date'].tail().tolist()}")
        else:
            print(f"\n{model}: No data found for gauge {gauge_id} and horizon {horizon}")
    else:
        print(f"\n{model}: File not found")

print("\n" + "-" * 80)
print("\nChecking overall statistics for all models...")

for model in models:
    parquet_file = cache_path / f"{model}.parquet"

    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)

        print(f"\n{model} overall stats:")
        print(f"  Total rows: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique gauges: {df['gauge_id'].nunique()}")
        print(f"  Unique horizons: {sorted(df['horizon'].unique())}")

        # Check specifically for the gauge
        gauge_df = df[df["gauge_id"] == gauge_id]
        if not gauge_df.empty:
            print(f"  Date range for {gauge_id}: {gauge_df['date'].min()} to {gauge_df['date'].max()}")
            print(f"  Number of predictions for {gauge_id}: {len(gauge_df)}")
        else:
            print(f"  No data for gauge {gauge_id}")

print("\n" + "-" * 80)
print("\nDetailed analysis of date differences...")

# Load all dataframes
all_dfs = {}
for model in models:
    parquet_file = cache_path / f"{model}.parquet"
    if parquet_file.exists():
        all_dfs[model] = pd.read_parquet(parquet_file)

# Compare date ranges for the specific gauge across models
if all_dfs:
    print(f"\nComparing date ranges for gauge {gauge_id}:")
    date_ranges = {}

    for model, df in all_dfs.items():
        gauge_df = df[(df["gauge_id"] == gauge_id) & (df["horizon"] == horizon)]
        if not gauge_df.empty:
            date_ranges[model] = {"min": gauge_df["date"].min(), "max": gauge_df["date"].max(), "count": len(gauge_df)}

    # Find the overall min and max dates
    if date_ranges:
        overall_min = min(dr["min"] for dr in date_ranges.values())
        overall_max = max(dr["max"] for dr in date_ranges.values())

        print(f"\nOverall date range across all models: {overall_min} to {overall_max}")

        # Check which models have different ranges
        for model, dr in date_ranges.items():
            if dr["min"] != overall_min or dr["max"] != overall_max:
                print(f"\nWARNING: {model} has different date range!")
                print(f"  Expected: {overall_min} to {overall_max}")
                print(f"  Actual: {dr['min']} to {dr['max']}")
                print(f"  Missing from start: {(dr['min'] - overall_min).days if dr['min'] > overall_min else 0} days")
                print(f"  Missing from end: {(overall_max - dr['max']).days if dr['max'] < overall_max else 0} days")
