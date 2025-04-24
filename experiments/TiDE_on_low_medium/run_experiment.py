import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
sys.path.append(str(src_path))

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger  # Added import
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # Added import

from src.hydro_forecasting.data.lazy_datamodule import HydroLazyDataModule
from src.hydro_forecasting.preprocessing.grouped import GroupedPipeline
from sklearn.pipeline import Pipeline
from src.hydro_forecasting.preprocessing.standard_scale import StandardScaleTransformer
from src.hydro_forecasting.data.caravanify_parquet import (
    CaravanifyParquet,
    CaravanifyParquetConfig,
)

from src.hydro_forecasting.models.tide import LitTiDE, TiDEConfig

# Removed unused import: from src.hydro_forecasting.model_evaluation.evaluators import TSForecastEvaluator
from src.hydro_forecasting.model_evaluation.hp_from_yaml import hp_from_yaml

# --- Configuration ---
EXPERIMENT_NAME = "tide_low_medium_influence"
BASE_OUTPUT_DIR = project_root / "experiments" / "TiDE_on_low_medium" / "output"
CHECKPOINT_DIR = BASE_OUTPUT_DIR / "checkpoints"
LOGS_DIR = BASE_OUTPUT_DIR / "logs"
YAML_PATH = project_root / "notebooks" / "tide.yaml"
MAX_EPOCHS = 100
BATCH_SIZE = 2048
NUM_WORKERS = 12
EARLY_STOPPING_PATIENCE = 10
SAVE_TOP_K = 1

# Create output directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Loading ---
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

basin_ids = []
discarded_ids = []  # Renamed for clarity

print("==========LOADING LOW AND MEDIUM HUMAN INFLUENCE BASINS==========")
for region in regions:
    # Construct paths relative to project root or use absolute paths as needed
    attributes_dir = (
        f"/Users/cooper/Desktop/CaravanifyParquet/{region}/post_processed/attributes"
    )
    timeseries_dir = f"/Users/cooper/Desktop/CaravanifyParquet/{region}/post_processed/timeseries/csv"
    # shapefile_dir = f"/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY/{region}/post_processed/shapefiles" # Unused
    human_influence_path = (
        project_root
        / "scripts"
        / "human_influence_index"
        / "results"
        / "human_influence_classification.parquet"
    )

    config = CaravanifyParquetConfig(
        attributes_dir=attributes_dir,
        timeseries_dir=timeseries_dir,
        # shapefile_dir=shapefile_dir, # Removed unused parameter
        human_influence_path=str(human_influence_path),  # Ensure path is string
        gauge_id_prefix=f"{region}",
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    caravan = CaravanifyParquet(config)
    region_basin_ids = caravan.get_all_gauge_ids()  # Use a temporary variable

    filtered_ids, current_discarded_ids = caravan.filter_gauge_ids_by_human_influence(
        region_basin_ids, ["Low", "Medium"]
    )

    basin_ids.extend(filtered_ids)
    discarded_ids.extend(current_discarded_ids)  # Append discarded IDs for this region

print(f"Total basins to process: {len(basin_ids)}")
print(f"Total discarded basins: {len(discarded_ids)}")


print("==========LOADING TiDE HYPERPARAMETERS==========")
# Use configured YAML_PATH
tide_hp = hp_from_yaml("tide", YAML_PATH)


print("=========SETTING UP LAZY DATA MODULE==========")
print("Defining features")
forcing_features = [
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
    "surface_net_thermal_radiation_mean",
    "potential_evaporation_sum_ERA5_LAND",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "total_precipitation_sum",
]

static_features = [
    "p_mean",
    "area",
    "ele_mt_sav",
    "high_prec_dur",
    "frac_snow",
    "high_prec_freq",
    "slp_dg_sav",
    "cly_pc_sav",
    "aridity_ERA5_LAND",
    "aridity_FAO_PM",
]

print("Defining pipeline")

feature_pipeline = GroupedPipeline(
    Pipeline([("scaler", StandardScaleTransformer())]),
    columns=forcing_features,
    group_identifier="gauge_id",
)

target_pipeline = GroupedPipeline(
    Pipeline([("scaler", StandardScaleTransformer())]),
    columns=["streamflow"],
    group_identifier="gauge_id",
)

static_pipeline = Pipeline([("scaler", StandardScaleTransformer())])

preprocessing_config = {
    "features": {"pipeline": feature_pipeline},
    "target": {"pipeline": target_pipeline},
    "static_features": {"pipeline": static_pipeline, "columns": static_features},
}

print("Defining region directory maps")
# Use absolute paths or paths relative to a known base
region_time_series_base_dirs = {
    region: f"/Users/cooper/Desktop/CaravanifyParquet/{region}/post_processed/timeseries/csv/{region}"
    for region in regions
}

region_static_attributes_base_dirs = {
    region: f"/Users/cooper/Desktop/CaravanifyParquet/{region}/post_processed/attributes/{region}"
    for region in regions
}

print("Defining data module")
# Define preprocessing output directory relative to BASE_OUTPUT_DIR
preprocessing_output_dir = BASE_OUTPUT_DIR / "preprocessing_cache"
preprocessing_output_dir.mkdir(parents=True, exist_ok=True)

datamodule = HydroLazyDataModule(
    region_time_series_base_dirs=region_time_series_base_dirs,
    region_static_attributes_base_dirs=region_static_attributes_base_dirs,
    path_to_preprocessing_output_directory=str(
        preprocessing_output_dir
    ),  # Use configured path
    group_identifier="gauge_id",
    batch_size=BATCH_SIZE,  # Use configured batch size
    input_length=tide_hp["input_len"],
    output_length=tide_hp["output_len"],
    forcing_features=forcing_features,
    static_features=static_features,
    target="streamflow",
    preprocessing_configs=preprocessing_config,
    num_workers=NUM_WORKERS,  # Use configured num workers
    min_train_years=5,
    train_prop=0.5,
    val_prop=0.25,
    test_prop=0.25,
    max_imputation_gap_size=5,
    list_of_gauge_ids_to_process=basin_ids,
    is_autoregressive=True,
    files_per_batch=150,
)


print("=========TRAINING THE MODEL==========")
print("Instantiating the model")

config = TiDEConfig(**tide_hp)
model = LitTiDE(config)

print("Defining the logger and callbacks")
# Setup TensorBoard Logger
logger = TensorBoardLogger(
    save_dir=str(LOGS_DIR),
    name=EXPERIMENT_NAME,
    version="run_0",  # Example versioning, adjust if running multiple times
)

# Setup Callbacks
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,  # Use configured patience
    verbose=True,
    mode="min",
)

# Setup Model Checkpoint to save in the configured directory
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(CHECKPOINT_DIR),  # Use configured checkpoint directory
    filename=f"{EXPERIMENT_NAME}-{{epoch:02d}}-{{val_loss:.4f}}",  # Consistent naming
    save_top_k=SAVE_TOP_K,  # Use configured save_top_k
    mode="min",
    save_last=True,  # Optionally save the last checkpoint
)

print("Defining the trainer")
trainer = pl.Trainer(
    accelerator="cuda"
    if pl.accelerators.cuda.is_available()
    else "cpu",  # Check availability
    devices=1,
    max_epochs=MAX_EPOCHS,  # Use configured max_epochs
    enable_progress_bar=True,
    logger=logger,  # Pass the logger instance
    callbacks=[
        early_stopping_callback,  # Use defined callback instance
        checkpoint_callback,  # Use defined callback instance
    ],
)
print("Training the model")
trainer.fit(
    model,
    datamodule=datamodule,
)

print(f"=========DONE==========")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
print(f"Logs saved to: {LOGS_DIR}/{EXPERIMENT_NAME}/run_0")
print(f"Best checkpoint path: {checkpoint_callback.best_model_path}")
