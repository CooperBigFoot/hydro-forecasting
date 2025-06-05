import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from src.hydro_forecasting.data.caravanify_parquet import CaravanifyParquet, CaravanifyParquetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Constants:
    """Constants used throughout the classification pipeline."""

    DEFAULT_RANDOM_STATE = 42
    MIN_CV_FOLDS = 5
    DEFAULT_N_ITER_SEARCH = 20

    DEFAULT_STATIC_FEATURES = [
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
    random_state: int = Constants.DEFAULT_RANDOM_STATE
    n_jobs: int = -1

    # Cross-validation parameters
    cv_folds: int = 5

    # Features to use for classification
    statics_to_keep: list[str] | None = None

    def __post_init__(self):
        """Validate configuration and set defaults."""
        self._validate_paths()
        self._validate_countries()
        self._validate_parameters()

        if self.statics_to_keep is None:
            self.statics_to_keep = Constants.DEFAULT_STATIC_FEATURES.copy()

    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        paths_to_check = [
            ("attributes_base_dir", self.attributes_base_dir),
            ("timeseries_base_dir", self.timeseries_base_dir),
            ("cluster_assignment_path", self.cluster_assignment_path),
        ]

        for name, path in paths_to_check:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Required path '{name}' does not exist: {path}")

    def _validate_countries(self) -> None:
        """Validate country configuration."""
        if not self.source_countries:
            raise ValueError("At least one source country must be specified")

        if not self.target_country:
            raise ValueError("Target country must be specified")

        if self.target_country in self.source_countries:
            logger.warning(f"Target country '{self.target_country}' is also in source countries")

    def _validate_parameters(self) -> None:
        """Validate model and data parameters."""
        if self.cv_folds < Constants.MIN_CV_FOLDS:
            raise ValueError(f"cv_folds must be at least {Constants.MIN_CV_FOLDS}")


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""

    pass


def validate_data_quality(df: pd.DataFrame, required_cols: list[str], name: str = "DataFrame") -> None:
    """
    Validate data quality and raise informative errors.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        name: Name of the dataset for error messages

    Raises:
        DataValidationError: If data quality issues are found
    """
    if df.empty:
        raise DataValidationError(f"{name} is empty")

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"{name} missing required columns: {missing_cols}")


def validate_feature_alignment(X_train: pd.DataFrame, X_target: pd.DataFrame) -> None:
    """
    Ensure training and target features are properly aligned.

    Args:
        X_train: Training feature DataFrame
        X_target: Target feature DataFrame
    """
    train_features = set(X_train.columns)
    target_features = set(X_target.columns)

    if train_features != target_features:
        missing_in_target = train_features - target_features
        extra_in_target = target_features - train_features

        if missing_in_target:
            logger.warning(f"Features missing in target data: {missing_in_target}")
        if extra_in_target:
            logger.warning(f"Extra features in target data: {extra_in_target}")


def get_cluster_assignments(config: ClassificationConfig) -> pd.DataFrame:
    """
    Load cluster assignments from CSV file.
    This should only be called ONCE at the beginning.

    Args:
        config: Classification configuration

    Returns:
        DataFrame with gauge_id and cluster assignments

    Raises:
        DataValidationError: If cluster assignment file is invalid
    """
    logger.info(f"Loading cluster assignments from {config.cluster_assignment_path}")

    try:
        cluster_assignments = pd.read_csv(config.cluster_assignment_path)
    except Exception as e:
        raise DataValidationError(f"Failed to load cluster assignments: {e}") from e

    # Validate required columns
    if "gauge_id" not in cluster_assignments.columns:
        raise DataValidationError("Missing 'gauge_id' column in cluster assignments file")

    # Extract country from gauge_id if not already present
    if "country" not in cluster_assignments.columns:
        cluster_assignments["country"] = cluster_assignments["gauge_id"].str.split("_").str[0]

    logger.info(f"Loaded {len(cluster_assignments)} cluster assignments")
    return cluster_assignments


def get_country_basins_from_cluster_assignments(country: str, cluster_assignments: pd.DataFrame) -> list[str]:
    """
    Extract unique basins for a given country from cluster assignments DataFrame.

    Args:
        country: Country code (e.g., 'CH', 'CL', 'USA')
        cluster_assignments: Pre-loaded cluster assignments DataFrame

    Returns:
        List of gauge_ids for the specified country
    """
    country_basins = cluster_assignments[cluster_assignments["country"] == country]["gauge_id"].unique().tolist()

    logger.info(f"Found {len(country_basins)} basins with cluster assignments for {country}")
    return country_basins


def map_clusters_to_basins(static_df: pd.DataFrame, cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Map clusters to basins for a given country.

    Args:
        static_df: DataFrame with static attributes
        cluster_assignments: Pre-loaded cluster assignments DataFrame

    Returns:
        DataFrame with added cluster column

    Raises:
        DataValidationError: If gauge_id column is missing
    """
    if "gauge_id" not in static_df.columns:
        raise DataValidationError("Static DataFrame must contain a 'gauge_id' column")

    # Create mapping from gauge_id to cluster
    cluster_map = dict(zip(cluster_assignments["gauge_id"], cluster_assignments["cluster"], strict=False))

    # Map gauge_id in static df to cluster in cluster assignments
    static_df = static_df.copy()  # Create a copy to avoid modifying the original
    static_df["cluster"] = static_df["gauge_id"].map(cluster_map)

    # Check if mapping was successful
    missing_clusters = static_df["cluster"].isna().sum()
    if missing_clusters > 0:
        logger.warning(f"{missing_clusters} basins have no cluster assignment")
        example_missing = static_df[static_df["cluster"].isna()]["gauge_id"].head().tolist()
        logger.warning(f"Example missing basins: {example_missing}")

    return static_df


def get_labelled_data_by_country(
    country: str, config: ClassificationConfig, cluster_assignments: pd.DataFrame
) -> pd.DataFrame:
    """
    Load and prepare data for a country.

    Args:
        country: Country code (e.g., 'CH', 'CL', 'USA')
        config: Classification configuration
        cluster_assignments: Pre-loaded cluster assignments DataFrame

    Returns:
        DataFrame with static attributes and cluster assignments
    """
    logger.info(f"Loading data for {country}...")

    try:
        # Get basin IDs with cluster assignments (no CSV read here!)
        basin_ids = get_country_basins_from_cluster_assignments(country, cluster_assignments)

        if not basin_ids:
            logger.warning(f"No basins with cluster assignments found for {country}")
            return pd.DataFrame()

        # Configure Caravanify
        attributes_path = Path(config.attributes_base_dir) / country / "post_processed" / "attributes"
        timeseries_path = Path(config.timeseries_base_dir) / country / "post_processed" / "timeseries" / "csv"

        caravan_config = CaravanifyParquetConfig(
            attributes_dir=str(attributes_path),
            timeseries_dir=str(timeseries_path),
            gauge_id_prefix=country,
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        caravan = CaravanifyParquet(caravan_config)

        # Load ONLY static attributes (skip expensive time series loading)
        caravan._validate_gauge_ids(basin_ids)
        caravan._load_static_attributes(basin_ids)
        static_df = caravan.get_static_attributes()

        if static_df.empty:
            logger.warning(f"No static attributes found for {country}")
            return pd.DataFrame()

        # Ensure we have all the required columns
        missing_cols = [col for col in config.statics_to_keep if col not in static_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in static data for {country}: {missing_cols}")
            # Add missing columns with NaN values
            for col in missing_cols:
                static_df[col] = np.nan

        # Get only the columns we want to keep
        static_df = static_df[config.statics_to_keep]

        # Add cluster assignments (no CSV read here!)
        labeled_df = map_clusters_to_basins(static_df, cluster_assignments)

        # Filter rows with missing cluster
        result_df = labeled_df.dropna(subset=["cluster"])

        logger.info(f"Processed {len(result_df)} basins with valid data and cluster assignments for {country}")
        return result_df

    except Exception as e:
        logger.error(f"Error loading data for {country}: {e}")
        return pd.DataFrame()


def get_unlabelled_data_for_target(config: ClassificationConfig) -> pd.DataFrame:
    """
    Load data for target country without cluster assignments.

    Args:
        config: Classification configuration

    Returns:
        DataFrame with static attributes for target country
    """
    logger.info(f"Loading data for target country {config.target_country}...")

    try:
        # Configure Caravanify for target country
        attributes_path = Path(config.attributes_base_dir) / config.target_country / "post_processed" / "attributes"
        timeseries_path = (
            Path(config.timeseries_base_dir) / config.target_country / "post_processed" / "timeseries" / "csv"
        )

        caravan_config = CaravanifyParquetConfig(
            attributes_dir=str(attributes_path),
            timeseries_dir=str(timeseries_path),
            gauge_id_prefix=config.target_country,
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        caravan = CaravanifyParquet(caravan_config)
        ids = caravan.get_all_gauge_ids()

        if not ids:
            logger.warning(f"No basins found for {config.target_country}")
            return pd.DataFrame()

        logger.info(f"Found {len(ids)} basins for {config.target_country}")

        # Load ONLY static attributes (skip expensive time series loading)
        caravan._validate_gauge_ids(ids)
        caravan._load_static_attributes(ids)
        static_df = caravan.get_static_attributes()

        if static_df.empty:
            logger.warning(f"No static attributes found for {config.target_country}")
            return pd.DataFrame()

        # Ensure we have all required columns
        missing_cols = [col for col in config.statics_to_keep if col not in static_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in target data: {missing_cols}")
            # Add missing columns with NaN values
            for col in missing_cols:
                static_df[col] = np.nan

        # Select only needed columns
        result_df = static_df[config.statics_to_keep]
        return result_df

    except Exception as e:
        logger.error(f"Error loading target data: {e}")
        return pd.DataFrame()


def create_mutual_info_heatmap(X: pd.DataFrame, output_path: Path) -> None:
    """
    Create and save mutual information heatmap with improved efficiency.

    Args:
        X: DataFrame with numeric features
        output_path: Path to save the heatmap
    """
    logger.info("Creating mutual information heatmap...")

    try:
        # Ensure we have numeric data
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            logger.warning("No numeric features available for mutual information analysis")
            return

        # Drop rows with any missing values for MI calculation
        X_numeric = X_numeric.dropna()
        if X_numeric.empty:
            logger.warning("No complete cases available for mutual information analysis")
            return

        features = X_numeric.columns

        # Initialize matrix with identity diagonal
        mi_matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)

        # Calculate only upper triangle for efficiency
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                try:
                    # Calculate MI between features i and j
                    mi_score = mutual_info_regression(
                        X_numeric.iloc[:, [i]], X_numeric.iloc[:, j], random_state=Constants.DEFAULT_RANDOM_STATE
                    )[0]

                    # Mirror to both sides of the matrix
                    mi_matrix.iloc[i, j] = mi_score
                    mi_matrix.iloc[j, i] = mi_score

                except Exception as e:
                    logger.warning(f"Error calculating MI between {features[i]} and {features[j]}: {e}")
                    mi_matrix.iloc[i, j] = 0
                    mi_matrix.iloc[j, i] = 0

        # Set diagonal to NaN for better visualization
        np.fill_diagonal(mi_matrix.values, np.nan)

        # Plot the symmetric mutual information heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(mi_matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Symmetric Mutual Information Heatmap")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved mutual information heatmap to {output_path}")

    except Exception as e:
        logger.error(f"Error creating mutual information heatmap: {e}")


def create_baseline_model(config: ClassificationConfig) -> RandomForestClassifier:
    """
    Create a baseline Random Forest model with default parameters.

    Args:
        config: Classification configuration

    Returns:
        RandomForestClassifier with default parameters
    """
    logger.info("Creating baseline model with default parameters")
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def train_rf_classifier(
    X: pd.DataFrame, y: pd.Series, config: ClassificationConfig
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """
    Train Random Forest classifier with hyperparameter tuning.

    Args:
        X: Feature DataFrame
        y: Target series
        config: Classification configuration

    Returns:
        Tuple of (best model, performance metrics)

    Raises:
        ModelTrainingError: If training fails
    """
    logger.info("Training Random Forest classifier...")

    try:
        # Ensure we have numeric data
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            raise ModelTrainingError("No numeric features available for training")

        # Drop rows with missing values
        valid_mask = X_numeric.notna().all(axis=1)
        X_clean = X_numeric[valid_mask]
        y_clean = y[valid_mask]

        if X_clean.empty:
            raise ModelTrainingError("No complete cases available for training")

        logger.info(f"Training with {len(X_clean)} complete cases out of {len(X_numeric)} total samples")

        # Validate that we have enough samples for cross-validation
        n_classes = len(np.unique(y_clean))
        cv_folds = min(config.cv_folds, n_classes)

        if cv_folds < Constants.MIN_CV_FOLDS:
            logger.warning(f"Using {cv_folds} CV folds due to limited classes")

        # Define cross-validation strategy
        cv = StratifiedKFold(
            n_splits=cv_folds,
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
            n_iter=Constants.DEFAULT_N_ITER_SEARCH,
            scoring="accuracy",
            cv=cv,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        rf_search.fit(X_clean, y_clean)
        best_rf = rf_search.best_estimator_

        # Cross-validation for prediction probabilities
        proba_predictions = cross_val_predict(best_rf, X_clean, y_clean, cv=cv, method="predict_proba")

        # Compute metrics
        log_loss_score = log_loss(y_clean, proba_predictions)
        accuracy = rf_search.best_score_

        metrics = {
            "best_params": rf_search.best_params_,
            "accuracy": accuracy,
            "log_loss": log_loss_score,
            "n_features": X_clean.shape[1],
            "n_samples": X_clean.shape[0],
            "n_classes": n_classes,
        }

        logger.info(f"Best parameters: {rf_search.best_params_}")
        logger.info(f"Cross-validation accuracy: {accuracy:.4f}")
        logger.info(f"Log loss score: {log_loss_score:.4f}")

        # Train final model on all clean data
        final_model = RandomForestClassifier(
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            **rf_search.best_params_,
        )
        final_model.fit(X_clean, y_clean)

        return final_model, metrics

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Fall back to default parameters
        try:
            X_numeric = X.select_dtypes(include=[np.number])
            valid_mask = X_numeric.notna().all(axis=1)
            X_clean = X_numeric[valid_mask]
            y_clean = y[valid_mask]

            if X_clean.empty:
                raise ModelTrainingError("No complete cases for fallback training")

            default_rf = create_baseline_model(config)
            default_rf.fit(X_clean, y_clean)

            metrics = {
                "best_params": {
                    "n_estimators": config.n_estimators,
                    "max_depth": config.max_depth,
                    "min_samples_split": config.min_samples_split,
                    "min_samples_leaf": config.min_samples_leaf,
                },
                "accuracy": np.nan,
                "log_loss": np.nan,
                "n_features": X_clean.shape[1],
                "n_samples": X_clean.shape[0],
                "n_classes": len(np.unique(y_clean)),
            }

            logger.info("Fallback to baseline model successful")
            return default_rf, metrics

        except Exception as fallback_error:
            raise ModelTrainingError(f"Both primary and fallback training failed: {fallback_error}") from fallback_error


def plot_feature_importance(model: RandomForestClassifier, feature_names: list[str], output_path: Path) -> None:
    """
    Plot and save feature importance from Random Forest.

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    logger.info("Creating feature importance plot...")

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
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature importance plot to {output_path}")

        # Log top 5 most important features
        top_features = [(sorted_features[i], sorted_importances[i]) for i in range(min(5, len(sorted_features)))]
        logger.info(f"Top 5 most important features: {top_features}")

    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")


def predict_target_clusters(
    model: RandomForestClassifier,
    X_target: pd.DataFrame,
    target_ids: pd.Series,
    output_path: Path,
    config: ClassificationConfig,
) -> pd.DataFrame:
    """
    Predict cluster probabilities for target country watersheds.

    Args:
        model: Trained Random Forest model
        X_target: Target features DataFrame
        target_ids: Target gauge IDs
        output_path: Path to save predictions
        config: Classification configuration

    Returns:
        DataFrame with predictions and probabilities
    """
    logger.info(f"Predicting clusters for {len(X_target)} target basins...")

    try:
        # Ensure we have numeric data
        X_numeric = X_target.select_dtypes(include=[np.number])
        if X_numeric.empty:
            raise ValueError("No numeric features available for prediction")

        # Only predict for complete cases
        valid_mask = X_numeric.notna().all(axis=1)
        X_clean = X_numeric[valid_mask]
        clean_ids = target_ids[valid_mask]

        if X_clean.empty:
            logger.warning("No complete cases available for prediction")
            return pd.DataFrame()

        logger.info(f"Predicting for {len(X_clean)} complete cases out of {len(X_numeric)} total targets")

        # Get cluster probabilities
        proba_predictions = model.predict_proba(X_clean)

        # Create DataFrame with gauge_id and probabilities for each class
        proba_df = pd.DataFrame(
            proba_predictions, columns=[f"cluster_{model.classes_[i]}_prob" for i in range(len(model.classes_))]
        )
        proba_df["gauge_id"] = clean_ids.values

        # Reorder columns to have gauge_id first
        cols = ["gauge_id"] + [col for col in proba_df.columns if col != "gauge_id"]
        proba_df = proba_df[cols]

        # Add predicted cluster (highest probability)
        predicted_clusters = model.predict(X_clean)
        proba_df["predicted_cluster"] = predicted_clusters

        # Add prediction confidence (max probability)
        proba_df["prediction_confidence"] = np.max(proba_predictions, axis=1)

        # Save to CSV
        proba_df.to_csv(output_path, index=False)
        logger.info(f"Saved cluster predictions to {output_path}")

        # Log prediction summary
        cluster_counts = pd.Series(predicted_clusters).value_counts()
        logger.info(f"Prediction summary: {cluster_counts.to_dict()}")
        logger.info(f"Average prediction confidence: {proba_df['prediction_confidence'].mean():.3f}")

        return proba_df

    except Exception as e:
        logger.error(f"Error predicting target clusters: {e}")
        return pd.DataFrame()


def load_and_combine_source_data(config: ClassificationConfig) -> pd.DataFrame:
    """
    Load and combine data from all source countries.
    OPTIMIZED: Load cluster assignments only once!

    Args:
        config: Classification configuration

    Returns:
        Combined DataFrame from all source countries

    Raises:
        DataValidationError: If no valid data is found
    """
    logger.info("Loading and combining source country data...")

    # LOAD CLUSTER ASSIGNMENTS ONLY ONCE!
    cluster_assignments = get_cluster_assignments(config)

    all_data = []
    for country in config.source_countries:
        try:
            # Pass the pre-loaded cluster assignments
            country_data = get_labelled_data_by_country(country, config, cluster_assignments)
            if not country_data.empty:
                country_data["source_country"] = country  # Track source country
                all_data.append(country_data)
                logger.info(f"Successfully loaded {len(country_data)} samples from {country}")
            else:
                logger.warning(f"No data loaded for {country}")
        except Exception as e:
            logger.error(f"Failed to load data for {country}: {e}")

    if not all_data:
        raise DataValidationError("No valid data from source countries. Cannot proceed.")

    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data from {len(all_data)} countries: {len(combined_data)} total samples")

    return combined_data


def prepare_training_data(combined_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector for training.

    Args:
        combined_data: Combined data from source countries

    Returns:
        Tuple of (features DataFrame, target Series)

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Preparing training data...")

    # Verify 'cluster' column exists
    if "cluster" not in combined_data.columns:
        available_cols = combined_data.columns.tolist()
        raise DataValidationError(f"'cluster' column not found. Available columns: {available_cols}")

    # Get feature columns (exclude gauge_id, cluster, and source_country)
    exclude_cols = {"gauge_id", "cluster", "source_country"}
    feature_columns = [col for col in combined_data.columns if col not in exclude_cols]

    X = combined_data[feature_columns]
    y = combined_data["cluster"]

    logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")

    # Log cluster distribution
    cluster_counts = y.value_counts().sort_index()
    logger.info("Cluster distribution:")
    for cluster, count in cluster_counts.items():
        logger.info(f"  Cluster {cluster}: {count} basins")

    return X, y


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series, config: ClassificationConfig) -> RandomForestClassifier:
    """
    Train and evaluate the Random Forest model.

    Args:
        X: Feature DataFrame
        y: Target Series
        config: Classification configuration

    Returns:
        Trained RandomForestClassifier
    """
    logger.info("Training and evaluating model...")

    # Validate data quality
    validate_data_quality(X, X.columns.tolist(), "Training features")

    # Train model
    model, metrics = train_rf_classifier(X, y, config)

    # Create visualizations
    output_dir = Path(config.output_dir)

    # Mutual information heatmap
    mi_heatmap_path = output_dir / "mutual_info_heatmap.png"
    create_mutual_info_heatmap(X, mi_heatmap_path)

    # Feature importance plot
    feature_importance_path = output_dir / "feature_importance.png"
    plot_feature_importance(model, X.columns.tolist(), feature_importance_path)

    # Save performance metrics
    metrics_path = output_dir / "performance_metrics.txt"
    save_performance_metrics(metrics, metrics_path)

    return model


def save_performance_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """
    Save performance metrics to a text file.

    Args:
        metrics: Dictionary of performance metrics
        output_path: Path to save the metrics
    """
    try:
        with open(output_path, "w") as f:
            f.write("Random Forest Classification Performance\n")
            f.write("=" * 45 + "\n\n")

            f.write("Model Configuration:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of features: {metrics.get('n_features', 'N/A')}\n")
            f.write(f"Number of samples: {metrics.get('n_samples', 'N/A')}\n")
            f.write(f"Number of classes: {metrics.get('n_classes', 'N/A')}\n\n")

            f.write("Best Hyperparameters:\n")
            f.write("-" * 20 + "\n")
            for param, value in metrics["best_params"].items():
                f.write(f"  {param}: {value}\n")

            f.write("\nPerformance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Cross-validation accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Log loss score: {metrics['log_loss']:.4f}\n")

        logger.info(f"Saved performance metrics to {output_path}")

    except Exception as e:
        logger.error(f"Error saving performance metrics: {e}")


def run_prediction_pipeline(config: ClassificationConfig, model: RandomForestClassifier) -> pd.DataFrame | None:
    """
    Run the prediction pipeline for the target country.

    Args:
        config: Classification configuration
        model: Trained RandomForestClassifier

    Returns:
        DataFrame with predictions, or None if prediction fails
    """
    logger.info("Running prediction pipeline...")

    try:
        # Load target country data
        target_data = get_unlabelled_data_for_target(config)

        if target_data.empty:
            logger.warning("No data available for target country. Skipping prediction.")
            return None

        # Prepare target data for prediction
        exclude_cols = {"gauge_id"}
        feature_columns = [col for col in target_data.columns if col not in exclude_cols]

        # Get features that were used in training
        training_features = model.feature_names_in_

        # Ensure feature alignment
        available_features = [col for col in training_features if col in feature_columns]
        missing_features = set(training_features) - set(available_features)

        if missing_features:
            logger.warning(f"Missing features in target data: {missing_features}")

        X_target = target_data[available_features].copy()

        # Add missing features with NaN (they will be handled by complete case analysis)
        for col in missing_features:
            X_target[col] = np.nan

        # Reorder columns to match training data
        X_target = X_target[training_features]
        target_ids = target_data["gauge_id"]

        # Validate feature alignment
        validate_feature_alignment(pd.DataFrame(columns=training_features), X_target)

        # Predict cluster probabilities
        output_dir = Path(config.output_dir)
        predictions_path = output_dir / "cluster_probabilities.csv"

        predictions = predict_target_clusters(model, X_target, target_ids, predictions_path, config)

        return predictions

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        return None


def run_training_pipeline(config: ClassificationConfig) -> RandomForestClassifier:
    """
    Run the complete training pipeline.

    Args:
        config: Classification configuration

    Returns:
        Trained RandomForestClassifier

    Raises:
        DataValidationError: If training data is invalid
        ModelTrainingError: If model training fails
    """
    logger.info("Running training pipeline...")

    # Load and combine source data
    combined_data = load_and_combine_source_data(config)

    # Prepare training data
    X, y = prepare_training_data(combined_data)

    # Train and evaluate model
    model = train_and_evaluate_model(X, y, config)

    return model


def main(config: ClassificationConfig) -> None:
    """
    Main function to run the Random Forest classification workflow.

    Args:
        config: Classification configuration
    """
    logger.info("Starting Random Forest classification workflow...")

    try:
        # Create output directory if it doesn't exist
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Run training pipeline
        model = run_training_pipeline(config)

        # Run prediction pipeline
        predictions = run_prediction_pipeline(config, model)

        if predictions is not None:
            logger.info("Classification workflow completed successfully!")
            logger.info(f"Predicted clusters for {len(predictions)} target basins")
        else:
            logger.warning("Classification workflow completed but predictions failed")

    except (DataValidationError, ModelTrainingError) as e:
        logger.error(f"Workflow failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in workflow: {e}")
        raise


if __name__ == "__main__":
    # Define configuration
    config = ClassificationConfig(
        attributes_base_dir="/workspace/CaravanifyParquet",
        timeseries_base_dir="/workspace/CaravanifyParquet",
        output_dir="/workspace/hydro-forecasting/scripts/cluster_basins/classification_results",
        cluster_assignment_path="/workspace/hydro-forecasting/scripts/cluster_basins/clustering_results/cluster_assignments_shifted_refactor.csv",
        source_countries=["CH", "CL", "USA", "camelsaus", "camelsgb", "camelsbr", "hysets", "lamah"],
        target_country="CA",
        cv_folds=10,
    )

    try:
        main(config)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
