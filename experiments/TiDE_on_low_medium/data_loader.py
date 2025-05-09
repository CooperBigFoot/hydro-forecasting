import logging
import sys
from pathlib import Path
from typing import Any  # Added Optional for type hint

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


def load_data(config: ExperimentConfig, project_root: Path, **cli_args: Any) -> list[str]:
    """
    Loads and filters basin/gauge IDs for the experiment.
    """
    logger.info("Starting basin/gauge ID loading and filtering process.")

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
        logger.debug(f"Added {src_path} to sys.path.")

    try:
        from hydro_forecasting.data.caravanify_parquet import (
            CaravanifyParquet,
            CaravanifyParquetConfig,
        )
    except ImportError as e:
        logger.error(f"Failed to import CaravanifyParquet: {e}. Ensure 'src' is in sys.path.")
        raise

    basin_ids: list[str] = []
    discarded_ids: list[str] = []

    human_influence_path_resolved: Path | None = None
    if config.human_influence_index_path:
        p_human_influence = Path(config.human_influence_index_path)
        if not p_human_influence.is_absolute():
            human_influence_path_resolved = (project_root / p_human_influence).resolve()
            logger.debug(f"Resolved relative human_influence_index_path to: {human_influence_path_resolved}")
        else:
            human_influence_path_resolved = p_human_influence

    if not config.caravan_regions_to_load:
        logger.warning("No regions specified in 'config.caravan_regions_to_load'. No basin IDs will be loaded.")
        return []

    for region in config.caravan_regions_to_load:
        logger.info(f"Processing region: {region}")
        attributes_dir_str = config.caravan_attributes_base_dir_template.format(region=region)
        timeseries_dir_str = config.caravan_timeseries_base_dir_template.format(region=region)

        caravan_module_config = CaravanifyParquetConfig(
            attributes_dir=str(Path(attributes_dir_str).resolve()),
            timeseries_dir=str(Path(timeseries_dir_str).resolve()),
            human_influence_path=str(human_influence_path_resolved) if human_influence_path_resolved else None,
            gauge_id_prefix=config.caravan_gauge_id_prefix_map.get(region, region),
            use_hydroatlas_attributes=config.use_hydroatlas_attributes,
            use_caravan_attributes=config.use_caravan_attributes,
            use_other_attributes=config.use_other_attributes,
        )

        try:
            caravan_instance = CaravanifyParquet(caravan_module_config)
            region_basin_ids = caravan_instance.get_all_gauge_ids()
            logger.debug(f"Region {region}: Found {len(region_basin_ids)} raw basin IDs.")

            if human_influence_path_resolved and human_influence_path_resolved.exists():
                filtered_ids, current_discarded_ids = caravan_instance.filter_gauge_ids_by_human_influence(
                    region_basin_ids, config.human_influence_filter_categories
                )
                logger.info(
                    f"Region {region}: Filtered to {len(filtered_ids)} basins by human influence "
                    f"({config.human_influence_filter_categories}). Discarded {len(current_discarded_ids)}."
                )
            else:
                if config.human_influence_index_path:
                    logger.warning(
                        f"Human influence file not found at {human_influence_path_resolved} (specified in config). "
                        f"Using all basins for region {region}."
                    )
                else:
                    logger.info(f"No human influence file specified. Using all basins for region {region}.")
                filtered_ids = region_basin_ids
                current_discarded_ids = []

            basin_ids.extend(filtered_ids)
            discarded_ids.extend(current_discarded_ids)
        except FileNotFoundError as e:
            logger.warning(f"Data directory not found for region {region}. Skipping. Error: {e}")
        except Exception as e:
            logger.error(f"Could not process region {region}. Error: {e}", exc_info=True)  # exc_info for traceback

    logger.info(f"Total basins to process after filtering across all regions: {len(basin_ids)}")
    logger.info(f"Total discarded basins (due to human influence filter or errors): {len(discarded_ids)}")

    if not basin_ids:
        logger.warning("No basin IDs found after processing all regions and filters. Returning empty list.")
        return []

    if "filter_by_prefix" in cli_args and cli_args["filter_by_prefix"]:
        prefix_filter = cli_args["filter_by_prefix"]
        logger.info(f"Applying additional CLI filter: only basins starting with '{prefix_filter}'")
        original_count = len(basin_ids)
        basin_ids = [bid for bid in basin_ids if bid.startswith(prefix_filter)]
        logger.info(f"Basin IDs after prefix filter '{prefix_filter}': {len(basin_ids)} (from {original_count})")

    unique_sorted_basin_ids = sorted(list(set(basin_ids)))
    logger.info(f"Returning {len(unique_sorted_basin_ids)} unique, sorted basin IDs.")
    return unique_sorted_basin_ids
