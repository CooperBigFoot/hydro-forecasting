import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
sys.path.append(str(src_path))

from sklearn.pipeline import Pipeline

from hydro_forecasting.preprocessing.grouped import GroupedPipeline
from hydro_forecasting.preprocessing.normalize import NormalizeTransformer
from hydro_forecasting.preprocessing.standard_scale import StandardScaleTransformer


@dataclass
class ExperimentConfig:
    """
    Configuration for a hydrological forecasting experiment.
    Preprocessing pipelines are defined in __post_init__ if not provided.
    """

    experiment_name: str = "MyRefactoredExperiment"
    base_output_dir: str | Path = "outputs"

    region_time_series_base_dirs: dict[str, str | Path] = field(default_factory=dict)
    region_static_attributes_base_dirs: dict[str, str | Path] = field(default_factory=dict)
    path_to_preprocessing_output_directory: str | Path = "preprocessing_cache"
    group_identifier: str = "gauge_id"

    min_train_years: float = 5.0
    train_prop: float = 0.5
    val_prop: float = 0.25
    test_prop: float = 0.25
    max_imputation_gap_size: int = 5

    chunk_size: int = 2000
    validation_chunk_size: int | None = 4000
    num_workers: int = 10
    batch_size: int = 2048
    max_epochs: int = 300
    reload_dataloaders_every_n_epochs: int = 1
    early_stopping_patience: int = 30
    save_top_k_checkpoints: int = 1

    forcing_features: list[str] = field(
        default_factory=lambda: [
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
    )
    static_features: list[str] = field(
        default_factory=lambda: [
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
    )
    target: str = "streamflow"
    is_autoregressive: bool = True

    # This will be populated in __post_init__ if not provided at instantiation.
    preprocessing_configs: dict[str, dict[str, Any]] | None = None

    default_checkpoint_load_base_dir: str | Path | None = None

    accelerator: str = "cuda"
    devices: int = 1
    torch_float32_matmul_precision: str = "medium"
    num_sanity_val_steps: int = 0

    domain_id: str = "multi-region-default"
    domain_type: str = "source"

    human_influence_index_path: str | Path | None = (
        "/Users/cooper/Desktop/hydro-forecasting/scripts/human_influence_index/results/human_influence_classification.parquet"
    )
    human_influence_filter_categories: list[str] = field(default_factory=lambda: ["Low", "Medium"])
    caravan_gauge_id_prefix_map: dict[str, str] = field(default_factory=dict)
    caravan_regions_to_load: list[str] = field(
        default_factory=lambda: [
            "CL",
            "CH",
            "USA",
            "camelsaus",
            "camelsgb",
            "camelsbr",
            "hysets",
            "lamah",
        ]
    )
    caravan_attributes_base_dir_template: str = "/workspace/CaravanifyParquet/{region}/post_processed/attributes"
    caravan_timeseries_base_dir_template: str = "/workspace/CaravanifyParquet/{region}/post_processed/timeseries/csv"

    use_hydroatlas_attributes: bool = True
    use_caravan_attributes: bool = True
    use_other_attributes: bool = True

    input_length: int | None = None
    output_length: int | None = None

    def __post_init__(self):
        # Path conversions
        if isinstance(self.base_output_dir, str):
            self.base_output_dir = Path(self.base_output_dir)
        if isinstance(self.path_to_preprocessing_output_directory, str):
            self.path_to_preprocessing_output_directory = Path(self.path_to_preprocessing_output_directory)
        if self.default_checkpoint_load_base_dir and isinstance(self.default_checkpoint_load_base_dir, str):
            self.default_checkpoint_load_base_dir = Path(self.default_checkpoint_load_base_dir)
        if self.human_influence_index_path and isinstance(self.human_influence_index_path, str):
            self.human_influence_index_path = Path(self.human_influence_index_path)

        self.region_time_series_base_dirs = {k: Path(v) for k, v in self.region_time_series_base_dirs.items()}
        self.region_static_attributes_base_dirs = {
            k: Path(v) for k, v in self.region_static_attributes_base_dirs.items()
        }

        if self.validation_chunk_size is None and self.chunk_size is not None:
            self.validation_chunk_size = self.chunk_size * 2

        if self.preprocessing_configs is None:
            feature_pipeline = GroupedPipeline(
                Pipeline([("scaler", StandardScaleTransformer()), ("normalizer", NormalizeTransformer())]),
                columns=self.forcing_features,
                group_identifier=self.group_identifier,
            )
            target_pipeline = GroupedPipeline(
                Pipeline([("scaler", StandardScaleTransformer()), ("normalizer", NormalizeTransformer())]),
                columns=[self.target],
                group_identifier=self.group_identifier,
            )
            static_pipeline = Pipeline([("scaler", StandardScaleTransformer())])

            self.preprocessing_configs = {
                "features": {"pipeline": feature_pipeline},
                "target": {"pipeline": target_pipeline},
                "static_features": {
                    "pipeline": static_pipeline,
                    "columns": self.static_features,
                },
            }
