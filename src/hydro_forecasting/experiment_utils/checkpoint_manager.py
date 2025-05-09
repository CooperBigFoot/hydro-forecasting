import logging
import re
from pathlib import Path

from returns.pipeline import is_successful
from returns.result import Failure, Result, Success, safe

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR_NAME = "checkpoints"
LOGS_DIR_NAME = "logs"
RUN_PREFIX = "run_"
ATTEMPT_PREFIX = "attempt_"
BEST_MODEL_INFO_FILE = "overall_best_model_info.txt"
CHECKPOINT_EXTENSION = ".ckpt"


def _parse_attempt_number(dir_name: str) -> int:
    """
    Parses the attempt number from a directory name (e.g., "attempt_2" -> 2).

    Args:
        dir_name: The name of the directory.

    Returns:
        The attempt number.

    Raises:
        ValueError: If the directory name is not in the expected format.
    """
    match = re.fullmatch(rf"{ATTEMPT_PREFIX}(\d+)", dir_name)
    if not match:
        raise ValueError(f"Directory name '{dir_name}' is not in '{ATTEMPT_PREFIX}N' format.")
    return int(match.group(1))


@safe
def _find_latest_attempt_dir_in_run(run_dir: Path) -> Path | None:
    """
    Finds the subdirectory attempt_<N> with the highest N within a run directory.

    Args:
        run_dir: Path to the run directory (e.g., .../run_0/).

    Returns:
        Path to the latest attempt directory, or None if no attempt directories found.
    """
    if not run_dir.is_dir():
        logger.warning(f"Run directory {run_dir} does not exist or is not a directory.")
        return None
    attempt_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith(ATTEMPT_PREFIX)]
    if not attempt_dirs:
        return None

    latest_attempt_num = -1
    latest_attempt_dir = None
    for attempt_dir in attempt_dirs:
        try:
            num = _parse_attempt_number(attempt_dir.name)
            if num > latest_attempt_num:
                latest_attempt_num = num
                latest_attempt_dir = attempt_dir
        except ValueError:
            logger.warning(f"Skipping directory {attempt_dir.name} as it's not a valid attempt format.")
            continue
    return latest_attempt_dir


def _determine_attempt_path(base_run_path: Path, override_previous_attempt: bool) -> Result[Path, str]:
    """
    Determines the specific attempt directory path.

    Args:
        base_run_path: The path to the run directory (e.g., .../run_X/).
        override_previous_attempt: If True, overrides the latest existing attempt
                                   or creates attempt_0 if none exist.
                                   If False, creates a new incremented attempt.

    Returns:
        A Result containing the Path to the attempt directory or an error string.
    """
    try:
        base_run_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return Failure(f"Could not create base run path directory {base_run_path}: {e}")

    latest_attempt_dir_result = _find_latest_attempt_dir_in_run(base_run_path)

    # Corrected check using is_successful
    if not is_successful(latest_attempt_dir_result):
        return Failure(f"Error finding latest attempt dir: {latest_attempt_dir_result.failure()}")

    latest_attempt_dir = latest_attempt_dir_result.unwrap()

    if override_previous_attempt:
        if latest_attempt_dir:
            attempt_path = latest_attempt_dir
            logger.info(f"Overriding latest existing attempt: {attempt_path}")
        else:
            attempt_path = base_run_path / f"{ATTEMPT_PREFIX}0"
            logger.info(f"No existing attempts to override, creating: {attempt_path}")
    else:
        if latest_attempt_dir:
            try:
                latest_attempt_num = _parse_attempt_number(latest_attempt_dir.name)
                current_attempt_num = latest_attempt_num + 1
            except ValueError as e:
                return Failure(str(e))
        else:
            current_attempt_num = 0
        attempt_path = base_run_path / f"{ATTEMPT_PREFIX}{current_attempt_num}"
        logger.info(f"Creating new attempt: {attempt_path}")

    try:
        attempt_path.mkdir(parents=True, exist_ok=True)
        return Success(attempt_path)
    except OSError as e:
        return Failure(f"Could not create attempt directory {attempt_path}: {e}")


def determine_output_run_attempt_path(
    base_model_output_dir: Path,
    run_index: int,
    override_previous_attempts: bool,
) -> Result[Path, str]:
    """
    Determines and creates the versioned output path for a specific run and attempt
    for checkpoints.
    """
    if not isinstance(base_model_output_dir, Path):
        return Failure("base_model_output_dir must be a Path object.")
    if not isinstance(run_index, int) or run_index < 0:
        return Failure("run_index must be a non-negative integer.")

    run_path = base_model_output_dir / f"{RUN_PREFIX}{run_index}"
    return _determine_attempt_path(run_path, override_previous_attempts)


def determine_log_run_attempt_path(
    base_model_log_dir: Path, run_index: int, override_previous_attempts: bool
) -> Result[Path, str]:
    """
    Determines and creates the versioned output path for a specific run and attempt
    for logs.
    """
    if not isinstance(base_model_log_dir, Path):
        return Failure("base_model_log_dir must be a Path object.")
    if not isinstance(run_index, int) or run_index < 0:
        return Failure("run_index must be a non-negative integer.")

    run_path = base_model_log_dir / f"{RUN_PREFIX}{run_index}"
    return _determine_attempt_path(run_path, override_previous_attempts)


@safe
def _find_checkpoint_in_attempt_dir(attempt_dir: Path) -> Path | None:
    """
    Finds the .ckpt file within a specific attempt directory.
    """
    if not attempt_dir.is_dir():
        raise ValueError(f"Path {attempt_dir} is not a directory.")

    ckpt_files = list(attempt_dir.glob(f"*{CHECKPOINT_EXTENSION}"))

    if not ckpt_files:
        return None
    if len(ckpt_files) == 1:
        return ckpt_files[0]

    non_last_ckpt_files = [f for f in ckpt_files if f.name != f"last{CHECKPOINT_EXTENSION}"]
    if non_last_ckpt_files:
        if len(non_last_ckpt_files) == 1:
            return non_last_ckpt_files[0]
        else:
            logger.warning(
                f"Multiple non-last checkpoint files found in {attempt_dir}: {non_last_ckpt_files}. "
                f"Returning the first one found sorted by name: {sorted(non_last_ckpt_files)[0]}."
            )
            return sorted(non_last_ckpt_files)[0]
    else:
        logger.warning(
            f"Only 'last.ckpt' or multiple 'last.ckpt' files found in {attempt_dir}. "
            f"Returning the first one found sorted by name: {sorted(ckpt_files)[0]}."
        )
        return sorted(ckpt_files)[0]


def get_checkpoint_path_to_load(
    base_checkpoint_load_dir: Path,
    model_type: str,
    select_overall_best: bool,
    specific_run_index: int | None = None,
    specific_attempt_index: int | None = None,
) -> Result[Path, str]:
    """
    Resolves the exact .ckpt file path to load from an experiment's outputs.
    """
    if not isinstance(base_checkpoint_load_dir, Path):
        return Failure("base_checkpoint_load_dir must be a Path object.")
    if not base_checkpoint_load_dir.is_dir():
        return Failure(
            f"Base checkpoint load directory does not exist or is not a directory: {base_checkpoint_load_dir}"
        )

    model_type_checkpoint_dir = base_checkpoint_load_dir / model_type
    if not model_type_checkpoint_dir.is_dir():
        return Failure(f"Model type directory not found or is not a directory: {model_type_checkpoint_dir}")

    if select_overall_best:
        info_file_path = model_type_checkpoint_dir / BEST_MODEL_INFO_FILE
        if not info_file_path.is_file():
            return Failure(f"Overall best model info file not found or is not a file: {info_file_path}")
        try:
            with open(info_file_path) as f:
                relative_ckpt_path_str = f.read().strip()
            if not relative_ckpt_path_str:
                return Failure(f"Overall best model info file is empty: {info_file_path}")

            checkpoint_file_path = (model_type_checkpoint_dir / relative_ckpt_path_str).resolve()
            if not checkpoint_file_path.is_file():
                return Failure(
                    f"Checkpoint file specified in {info_file_path} does not exist or is not a file: {checkpoint_file_path}"
                )
            return Success(checkpoint_file_path)
        except OSError as e:
            return Failure(f"Error reading {info_file_path}: {e}")
        except Exception as e:
            return Failure(f"An unexpected error occurred while processing {info_file_path}: {e}")

    if specific_run_index is None:
        return Failure("specific_run_index must be provided if select_overall_best is False.")
    if not isinstance(specific_run_index, int) or specific_run_index < 0:
        return Failure("specific_run_index must be a non-negative integer.")

    run_dir = model_type_checkpoint_dir / f"{RUN_PREFIX}{specific_run_index}"
    if not run_dir.is_dir():
        return Failure(f"Run directory not found or is not a directory: {run_dir}")

    attempt_dir_to_load: Path | None
    if specific_attempt_index is not None:
        if not isinstance(specific_attempt_index, int) or specific_attempt_index < 0:
            return Failure("specific_attempt_index must be a non-negative integer.")
        attempt_dir_to_load = run_dir / f"{ATTEMPT_PREFIX}{specific_attempt_index}"
        if not attempt_dir_to_load.is_dir():
            return Failure(f"Specific attempt directory not found or is not a directory: {attempt_dir_to_load}")
    else:
        latest_attempt_dir_result = _find_latest_attempt_dir_in_run(run_dir)
        # Corrected check using is_successful
        if not is_successful(latest_attempt_dir_result):
            return Failure(f"Failed to query latest attempt for {run_dir}: {latest_attempt_dir_result.failure()}")

        latest_attempt_dir = latest_attempt_dir_result.unwrap()
        if not latest_attempt_dir:
            return Failure(f"No attempts found in run directory: {run_dir}")
        attempt_dir_to_load = latest_attempt_dir

    if not attempt_dir_to_load:
        return Failure(f"Could not determine attempt directory for run {specific_run_index}")

    find_ckpt_result = _find_checkpoint_in_attempt_dir(attempt_dir_to_load)
    # Corrected check using is_successful
    if not is_successful(find_ckpt_result):
        return Failure(f"Failed to query checkpoint in {attempt_dir_to_load}: {find_ckpt_result.failure()}")

    found_ckpt = find_ckpt_result.unwrap()
    if not found_ckpt:
        return Failure(f"No checkpoint file found in {attempt_dir_to_load}")
    return Success(found_ckpt.resolve())


def update_overall_best_model_info_file(
    model_checkpoints_output_dir: Path, best_checkpoint_relative_path: str
) -> Result[None, str]:
    """
    Writes or overwrites the `overall_best_model_info.txt` file for a model_type.
    """
    if not isinstance(model_checkpoints_output_dir, Path):
        return Failure("model_checkpoints_output_dir must be a Path object.")
    if not isinstance(best_checkpoint_relative_path, str):
        return Failure("best_checkpoint_relative_path must be a string.")

    try:
        model_checkpoints_output_dir.mkdir(parents=True, exist_ok=True)
        info_file_path = model_checkpoints_output_dir / BEST_MODEL_INFO_FILE

        full_referred_path = (model_checkpoints_output_dir / best_checkpoint_relative_path).resolve()
        if not full_referred_path.is_file():
            err_msg = f"Referred best checkpoint is not a file or does not exist: {full_referred_path}"
            logger.error(err_msg)
            return Failure(err_msg)

        with open(info_file_path, "w") as f:
            f.write(best_checkpoint_relative_path)
        logger.info(
            f"Updated {BEST_MODEL_INFO_FILE} at {model_checkpoints_output_dir} "
            f"to point to: {best_checkpoint_relative_path}"
        )
        return Success(None)
    except OSError as e:
        return Failure(f"Could not write to {BEST_MODEL_INFO_FILE} in {model_checkpoints_output_dir}: {e}")
    except Exception as e:
        return Failure(f"An unexpected error occurred while updating {BEST_MODEL_INFO_FILE}: {e}")
