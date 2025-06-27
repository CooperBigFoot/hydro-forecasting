import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore


def prepare_timeseries_data(
    df: pd.DataFrame,
    hemisphere_map: dict,
    basin_id_col: str = "gauge_id",
    date_col: str = "date",
    flow_col: str = "streamflow",
    standardize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Prepare time series data for clustering by aggregating to weekly data and standardizing.
    Excludes any basin with NaN values in the final standardized data.

    Args:
        df: DataFrame with daily streamflow data.
        hemisphere_map: Dictionary mapping gauge_id prefixes (string before '_')
                        to hemisphere strings ('southern' or 'northern').
        basin_id_col: Column name for basin ID.
        date_col: Column name for date.
        flow_col: Column name for streamflow.
        standardize: Whether to apply z-score standardization.

    Returns:
        Tuple containing:
        - Array of (standardized) weekly time series (shape: n_basins x 52)
        - List of basin IDs corresponding to the time series.
    """
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Extract gauge prefix from basin id and determine hemisphere, defaulting to 'northern'
    df["prefix"] = df[basin_id_col].str.split("_").str[0]
    df["hemisphere"] = df["prefix"].map(hemisphere_map).fillna("northern")

    month = df[date_col].dt.month
    year = df[date_col].dt.year

    # For southern hemisphere: water year starts on April 1
    southern_wy = np.where(month >= 4, year, year - 1)
    northern_wy = np.where(month < 10, year, year + 1)

    # Assign water_year based on hemisphere
    df["water_year"] = np.where(df["hemisphere"] == "southern", southern_wy, northern_wy)

    # Add week of water year (1-52) by computing days from start of water year
    df["day_of_water_year"] = df.groupby([basin_id_col, "water_year"])[date_col].transform(
        lambda x: (x - x.min()).dt.days
    )
    df["week"] = (df["day_of_water_year"] // 7) + 1
    df["week"] = df["week"].clip(upper=52)  # Ensure we don't exceed 52 weeks

    # Group by basin, water year and week, calculate mean weekly flow
    weekly_df = df.groupby([basin_id_col, "water_year", "week"])[flow_col].mean().reset_index()

    # Calculate mean annual cycle (52 weeks) for each basin
    mean_annual_df = weekly_df.groupby([basin_id_col, "week"])[flow_col].mean().reset_index()

    # Reshape to wide format (each row is a basin, each column is a week)
    wide_df = mean_annual_df.pivot(index=basin_id_col, columns="week", values=flow_col)

    # Ensure all 52 weeks are present
    for week in range(1, 53):
        if week not in wide_df.columns:
            wide_df[week] = np.nan
    wide_df = wide_df.reindex(columns=range(1, 53))

    basin_ids = wide_df.index.tolist()
    ts_data = []
    valid_basin_ids = []

    # Process each basin and filter out any with NaNs in final result
    for basin_id in basin_ids:
        series = wide_df.loc[basin_id].values

        # Fill any missing values with linear interpolation
        try:
            interpolated = pd.Series(series).interpolate().values

            # If we still have NaNs after interpolation, skip this basin
            if np.isnan(interpolated).any():
                continue

            # Apply standardization if requested
            if standardize:
                std_series = zscore(interpolated)
                # Check for NaNs after standardization (e.g., constant series with std=0)
                if np.isnan(std_series).any():
                    continue
                ts_data.append(std_series)
            else:
                ts_data.append(interpolated)

            # Only add basin_id if we successfully processed its data
            valid_basin_ids.append(basin_id)

        except Exception:
            # Skip basins with any errors during processing
            continue

    # Print diagnostics about filtering
    orig_count = len(basin_ids)
    final_count = len(valid_basin_ids)
    print(f"Started with {orig_count} basins, removed {orig_count - final_count} with NaNs, kept {final_count}")

    # Return only the valid, NaN-free data
    if final_count == 0:
        raise ValueError("No valid basins remained after filtering NaNs")

    return np.array(ts_data), valid_basin_ids


def plot_standardized_hydrographs(
    ts_data: np.ndarray,
    basin_ids: list[str],
    selected_basins: list[str] = None,
    max_display: int = 5,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot standardized weekly hydrographs for selected basins.

    Args:
        ts_data: Array of standardized weekly time series (shape: n_basins x 52)
        basin_ids: List of basin IDs corresponding to the time series
        selected_basins: List of basin IDs to plot (if None, will plot first max_display)
        max_display: Maximum number of basins to display
        figsize: Figure size (width, height)
    """
    basin_id_map = {id: i for i, id in enumerate(basin_ids)}

    if selected_basins is None:
        indices = list(range(min(len(basin_ids), max_display)))
        selected_basins = [basin_ids[i] for i in indices]
    else:
        indices = [basin_id_map[basin] for basin in selected_basins if basin in basin_id_map]
        selected_basins = [basin_ids[i] for i in indices]

    plt.figure(figsize=figsize)
    weeks = np.arange(1, 53)

    # Generate colors from a colormap for distinct lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))

    for _i, (idx, basin, color) in enumerate(zip(indices, selected_basins, colors, strict=False)):
        plt.plot(weeks, ts_data[idx], label=f"Basin {basin}", color=color, linewidth=2)

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    plt.xlabel("Week of Water Year")
    plt.ylabel("Standardized Flow (Z-score)")
    plt.title("Standardized Weekly Hydrographs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.show()
