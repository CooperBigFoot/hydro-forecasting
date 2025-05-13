import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import log_loss
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_models.caravanify import Caravanify, CaravanifyConfig


@dataclass
class ClassificationConfig:
    """Configuration for the time series clustering process."""

    # Data directories
    attributes_base_dir: str
    timeseries_base_dir: str

    # Output configuration
    output_dir: str
    cluster_assignment_path: str

    # Source countries (with labeled data) and target country (to predict)
    source_countries: list[str]
    target_country: str = "CA"

    # Random Forest parameters
    n_estimators: int = 200
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1

    # Cross-validation parameters
    cv_folds: int = 5

    # Features to use for classification
    statics_to_keep: list[str] = None

    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.statics_to_keep is None:
            self.statics_to_keep = [
                "gauge_id",
                "area",
                "gauge_lat",
                "gauge_lon",
                "frac_snow",
                "p_mean",
                "pet_mean_ERA5_LAND",
                "seasonality_ERA5_LAND",
                "aridity_ERA5_LAND",
                "slp_dg_sav",
                "high_prec_dur",
                "high_prec_freq",
                "low_prec_dur",
                "low_prec_freq",
                "cmi_ix_syr",
                "for_pc_sse",
                "glc_cl_smj",
                "rdd_mk_sav",
            ]


def get_cluster_assignments(config: ClassificationConfig) -> pd.DataFrame:
    """
    Load cluster assignments from CSV file.

    Args:
        config: Classification configuration

    Returns:
        DataFrame with gauge_id and cluster assignments
    """
    print(f"Loading cluster assignments from {config.cluster_assignment_path}")
    cluster_assignments = pd.read_csv(config.cluster_assignment_path)

    # Ensure we have 'gauge_id' column
    if "gauge_id" not in cluster_assignments.columns:
        raise ValueError("Missing 'gauge_id' column in cluster assignments file")

    # Extract country from gauge_id if not already present
    if "country" not in cluster_assignments.columns:
        cluster_assignments["country"] = cluster_assignments["gauge_id"].str.split("_").str[0]

    return cluster_assignments


def get_country_basins_from_cluster_assignments(country: str, config: ClassificationConfig) -> list[str]:
    """
    Extract unique basins for a given country from cluster assignments DataFrame.

    Args:
        country: Country code (e.g., 'CH', 'CL', 'USA')
        config: Classification configuration

    Returns:
        List of gauge_ids for the specified country
    """
    cluster_assignments = get_cluster_assignments(config)
    country_basins = cluster_assignments[cluster_assignments["country"] == country]["gauge_id"].unique().tolist()
    print(f"Found {len(country_basins)} basins with cluster assignments for {country}")
    return country_basins


def map_clusters_to_basins(static_df: pd.DataFrame, config: ClassificationConfig) -> pd.DataFrame:
    """
    Map clusters to basins for a given country.

    Args:
        static_df: DataFrame with static attributes
        config: Classification configuration

    Returns:
        DataFrame with added cluster column
    """
    if "gauge_id" not in static_df.columns:
        raise ValueError("Static DataFrame must contain a 'gauge_id' column")

    cluster_assignments = get_cluster_assignments(config)

    # Create mapping from gauge_id to cluster
    cluster_map = dict(zip(cluster_assignments["gauge_id"], cluster_assignments["cluster"], strict=False))

    # Map gauge_id in static df to cluster in cluster assignments
    static_df = static_df.copy()  # Create a copy to avoid modifying the original
    static_df["cluster"] = static_df["gauge_id"].map(cluster_map)

    # Check if mapping was successful
    missing_clusters = static_df["cluster"].isna().sum()
    if missing_clusters > 0:
        print(f"Warning: {missing_clusters} basins have no cluster assignment")
        print(f"Example missing basins: {static_df[static_df['cluster'].isna()]['gauge_id'].head().tolist()}")

    return static_df


def get_labelled_data_by_country(country: str, config: ClassificationConfig) -> pd.DataFrame:
    """
    Load and prepare data for a country.

    Args:
        country: Country code (e.g., 'CH', 'CL', 'USA')
        config: Classification configuration

    Returns:
        DataFrame with static attributes and cluster assignments
    """
    print(f"Loading data for {country}...")

    # Get basin IDs with cluster assignments
    basin_ids = get_country_basins_from_cluster_assignments(country, config)

    if not basin_ids:
        print(f"Warning: No basins with cluster assignments found for {country}")
        return pd.DataFrame()

    # Configure Caravanify
    caravan_config = CaravanifyConfig(
        attributes_dir=f"{config.attributes_base_dir}/{country}/post_processed/attributes",
        timeseries_dir=f"{config.timeseries_base_dir}/{country}/post_processed/timeseries/csv",
        gauge_id_prefix=country,
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    caravan = Caravanify(caravan_config)

    # Load stations
    caravan.load_stations(basin_ids)
    static_df = caravan.get_static_attributes()

    if static_df.empty:
        print(f"Warning: No static attributes found for {country}")
        return pd.DataFrame()

    # Ensure we have all the required columns
    missing_cols = [col for col in config.statics_to_keep if col not in static_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in static data for {country}: {missing_cols}")
        # Add missing columns with NaN values
        for col in missing_cols:
            static_df[col] = np.nan

    # Get only the columns we want to keep
    static_df = static_df[config.statics_to_keep]

    # Add cluster assignments - this will add the cluster column
    labeled_df = map_clusters_to_basins(static_df, config)

    # Filter rows with missing cluster
    result_df = labeled_df.dropna(subset=["cluster"])

    print(f"  Processed {len(result_df)} basins with valid data and cluster assignments for {country}")

    return result_df


def get_unlabelled_data_for_target(config: ClassificationConfig) -> pd.DataFrame:
    """
    Load data for target country without cluster assignments.

    Args:
        config: Classification configuration

    Returns:
        DataFrame with static attributes for target country
    """
    print(f"Loading data for target country {config.target_country}...")

    # Configure Caravanify for target country
    caravan_config = CaravanifyConfig(
        attributes_dir=f"{config.attributes_base_dir}/{config.target_country}/post_processed/attributes",
        timeseries_dir=f"{config.timeseries_base_dir}/{config.target_country}/post_processed/timeseries/csv",
        gauge_id_prefix=config.target_country,
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    caravan = Caravanify(caravan_config)
    ids = caravan.get_all_gauge_ids()

    if not ids:
        print(f"Warning: No basins found for {config.target_country}")
        return pd.DataFrame()

    print(f"  Got {len(ids)} basins for {config.target_country}")

    caravan.load_stations(ids)
    static_df = caravan.get_static_attributes()

    if static_df.empty:
        print(f"Warning: No static attributes found for {config.target_country}")
        return pd.DataFrame()

    # Ensure we have all required columns
    missing_cols = [col for col in config.statics_to_keep if col not in static_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in target data: {missing_cols}")
        # Add missing columns with NaN values
        for col in missing_cols:
            static_df[col] = np.nan

    # Select only needed columns
    result_df = static_df[config.statics_to_keep]

    return result_df


def create_mutual_info_heatmap(X: pd.DataFrame, output_path: str) -> None:
    """
    Create and save mutual information heatmap.

    Args:
        X: DataFrame with numeric features
        output_path: Path to save the heatmap
    """
    print("Creating mutual information heatmap...")

    # Ensure we have numeric data
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        print("Warning: No numeric features available for mutual information analysis")
        return

    features = X_numeric.columns

    # Initialize an empty DataFrame for mutual information
    mi_matrix = pd.DataFrame(np.zeros((len(features), len(features))), index=features, columns=features)

    # Compute pairwise mutual information and average the scores
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            if i == j:
                mi_matrix.loc[feature_i, feature_j] = np.nan
            else:
                try:
                    # MI: feature_i as predictor, feature_j as target
                    mi_ij = mutual_info_regression(X_numeric, X_numeric[feature_j], random_state=0)[i]
                    # MI: feature_j as predictor, feature_i as target
                    mi_ji = mutual_info_regression(X_numeric, X_numeric[feature_i], random_state=0)[j]
                    # Average to enforce symmetry
                    mi_matrix.loc[feature_i, feature_j] = (mi_ij + mi_ji) / 2
                except Exception as e:
                    print(f"Error calculating MI between {feature_i} and {feature_j}: {e}")
                    mi_matrix.loc[feature_i, feature_j] = np.nan

    # Plot the symmetric mutual information heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(mi_matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Symmetric Mutual Information Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved mutual information heatmap to {output_path}")


def train_rf_classifier(
    X: pd.DataFrame, y: pd.Series, config: ClassificationConfig
) -> tuple[RandomForestClassifier, dict]:
    """
    Train Random Forest classifier with hyperparameter tuning.

    Args:
        X: Feature DataFrame
        y: Target series
        config: Classification configuration

    Returns:
        Tuple of (best model, performance metrics)
    """
    print("Training Random Forest classifier...")

    # Ensure we have numeric data
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        raise ValueError("No numeric features available for training")

    # Ensure there are no missing values
    if X_numeric.isna().any().any():
        print("Warning: Missing values detected in features, filling with median values")
        X_numeric = X_numeric.fillna(X_numeric.median())

    # Define cross-validation strategy
    cv = StratifiedKFold(
        n_splits=min(config.cv_folds, len(np.unique(y))),
        shuffle=True,
        random_state=config.random_state,
    )

    # Define parameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Randomized Search
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=config.random_state, n_jobs=config.n_jobs),
        param_distributions=param_grid,
        n_iter=10,
        cv=cv,
        scoring="accuracy",
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    try:
        rf_search.fit(X_numeric, y)
        best_rf = rf_search.best_estimator_

        # Cross-validation for prediction probabilities
        proba_predictions = cross_val_predict(best_rf, X_numeric, y, cv=cv, method="predict_proba")

        # Compute metrics
        log_loss_score = log_loss(y, proba_predictions)
        accuracy = rf_search.best_score_

        metrics = {
            "best_params": rf_search.best_params_,
            "accuracy": accuracy,
            "log_loss": log_loss_score,
        }

        print(f"Best parameters: {rf_search.best_params_}")
        print(f"Cross-validation accuracy: {accuracy:.4f}")
        print(f"Log loss score: {log_loss_score:.4f}")

        # Train final model on all data
        final_model = RandomForestClassifier(
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            **rf_search.best_params_,
        )
        final_model.fit(X_numeric, y)

        return final_model, metrics

    except Exception as e:
        print(f"Error during model training: {e}")
        # Fall back to default parameters
        default_rf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        default_rf.fit(X_numeric, y)

        metrics = {
            "best_params": {
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "min_samples_split": config.min_samples_split,
                "min_samples_leaf": config.min_samples_leaf,
            },
            "accuracy": np.nan,
            "log_loss": np.nan,
        }

        return default_rf, metrics


def plot_feature_importance(model: RandomForestClassifier, feature_names: list[str], output_path: str) -> None:
    """
    Plot and save feature importance from Random Forest.

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    print("Creating feature importance plot...")
    try:
        importances = model.feature_importances_

        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.barh(sorted_features, sorted_importances, color="skyblue")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Random Forest Feature Importance")
        plt.gca().invert_yaxis()  # Invert y-axis to have most important feature on top
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved feature importance plot to {output_path}")
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")


def predict_target_clusters(
    model: RandomForestClassifier,
    X_target: pd.DataFrame,
    target_ids: pd.Series,
    output_path: str,
) -> pd.DataFrame:
    """
    Predict cluster probabilities for target country watersheds.
    """
    print(f"Predicting clusters for {len(X_target)} target basins...")

    # Ensure we have numeric data
    X_numeric = X_target.select_dtypes(include=[np.number])
    if X_numeric.empty:
        raise ValueError("No numeric features available for prediction")

    # Handle missing values
    if X_numeric.isna().any().any():
        print("Warning: Missing values detected in target features, filling with median values")
        X_numeric = X_numeric.fillna(X_numeric.median())

    try:
        # Get cluster probabilities
        proba_predictions = model.predict_proba(X_numeric)

        # Create DataFrame with gauge_id and probabilities for each class
        # Use the actual class labels from model.classes_ instead of indices
        proba_df = pd.DataFrame(
            proba_predictions,
            columns=[f"cluster_{model.classes_[i]}_prob" for i in range(len(model.classes_))],
        )
        proba_df["gauge_id"] = target_ids.values

        # Reorder columns to have gauge_id first
        cols = ["gauge_id"] + [col for col in proba_df.columns if col != "gauge_id"]
        proba_df = proba_df[cols]

        # Add predicted cluster (highest probability) - map back to original class labels
        predicted_clusters = model.predict(X_numeric)
        # This maps the internal indices back to the original class labels
        proba_df["predicted_cluster"] = predicted_clusters

        # Save to CSV
        proba_df.to_csv(output_path, index=False)
        print(f"Saved cluster predictions to {output_path}")

        return proba_df

    except Exception as e:
        print(f"Error predicting target clusters: {e}")
        return pd.DataFrame()


def main(config: ClassificationConfig) -> None:
    """
    Main function to run the Random Forest classification workflow.

    Args:
        config: Classification configuration
    """
    # Create output directory if it doesn't exist
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    mi_heatmap_path = output_dir / "mutual_info_heatmap.png"
    feature_importance_path = output_dir / "feature_importance.png"
    predictions_path = output_dir / "cluster_probabilities_shifted_15_clusters.csv"
    metrics_path = output_dir / "performance_metrics_shifted_15_clusters.txt"

    # Load and combine data from source countries
    all_data = []
    for country in config.source_countries:
        country_data = get_labelled_data_by_country(country, config)
        if not country_data.empty:
            all_data.append(country_data)

    if not all_data:
        print("Error: No valid data from source countries. Cannot proceed.")
        return

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data from {len(all_data)} countries: {len(combined_data)} rows")

    # Verify 'cluster' column exists
    if "cluster" not in combined_data.columns:
        print(
            f"Error: 'cluster' column not found in combined data. Available columns: {combined_data.columns.tolist()}"
        )
        return

    # Prepare training data
    # Get feature columns (all columns except gauge_id and cluster)
    feature_columns = [col for col in combined_data.columns if col not in ["gauge_id", "cluster"]]
    X = combined_data[feature_columns]
    y = combined_data["cluster"]

    # Print cluster distribution
    print("Cluster distribution:")
    for cluster, count in y.value_counts().items():
        print(f"  Cluster {cluster}: {count} basins")

    # Create mutual information heatmap
    create_mutual_info_heatmap(X, mi_heatmap_path)

    # Train Random Forest classifier
    rf_model, metrics = train_rf_classifier(X, y, config)

    # Plot feature importance
    plot_feature_importance(rf_model, X.columns.tolist(), feature_importance_path)

    # Save performance metrics
    with open(metrics_path, "w") as f:
        f.write("Random Forest Classification Performance\n")
        f.write("----------------------------------------\n")
        f.write("Best Parameters:\n")
        for param, value in metrics["best_params"].items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nCross-validation accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Log loss score: {metrics['log_loss']:.4f}\n")

    # Load target country data
    target_data = get_unlabelled_data_for_target(config)

    if target_data.empty:
        print("Warning: No data available for target country. Skipping prediction.")
        return

    # Prepare target data for prediction
    target_feature_columns = [col for col in feature_columns if col in target_data.columns]
    X_target = target_data[target_feature_columns]

    # Fill any missing columns that were in training but not in target data
    missing_cols = set(X.columns) - set(X_target.columns)
    for col in missing_cols:
        X_target[col] = 0  # Fill with zeros for missing columns

    X_target = X_target[X.columns]  # Reorder columns to match training data
    target_ids = target_data["gauge_id"]

    # Predict cluster probabilities for target country
    predict_target_clusters(rf_model, X_target, target_ids, predictions_path)

    print("Classification workflow completed successfully!")


if __name__ == "__main__":
    # Define configuration
    config = ClassificationConfig(
        attributes_base_dir="/workspace/CARAVANIFY",
        timeseries_base_dir="/workspace/CARAVANIFY",
        output_dir="./classification_results",
        cluster_assignment_path="/workspace/CAMELS-CH/clustering_results/cluster_assignments_shifted_refactor.csv",
        source_countries=["CH", "CL", "USA"],
        target_country="CA",
    )

    main(config)
