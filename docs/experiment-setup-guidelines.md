# Experiment Setup Guidelines for Hydrological Forecasting

## 1. Overview

This document outlines the standardized structure and conventions for setting up and running hydrological forecasting experiments. Adherence to these guidelines ensures reproducibility, simplifies collaboration, facilitates automation, and allows for easier integration with LLM-assisted development. The framework is designed to be flexible enough to accommodate various experiment types, including training from scratch, fine-tuning, and evaluating multiple models sequentially, while robustly managing experiment execution "attempts" (re-runs) and versioning of results.

## 2. Core Principles

* **Self-Contained Experiments**: Each experiment resides in its own directory (`experiments/YourExperimentName/`) and manages its specific configurations.
* **Centralized Logic**: Common functionalities like model instantiation, checkpoint fetching, data module configuration, and output path management are handled by modules in the `src` directory (e.g., `src/models/model_factory.py`, `src/experiment_utils/checkpoint_manager.py`).
* **Configuration over Code**: Experiment parameters are primarily defined in configuration files (`config.py` for experiment-level settings, YAML files for base model hyperparameters) rather than being hardcoded.
* **Standardized Interfaces**: Key components like data loaders and model factories follow consistent function signatures.
* **Consistent and Versioned Output Structure**: All experiments produce outputs in a predictable directory structure that supports versioning for different execution "attempts" of a logical run. This enables automated processing and reliable checkpoint reuse.
* **Modularity**: Clear separation of concerns between data loading logic (`data_loader.py`), experiment-specific configuration (`config.py`), the main experiment orchestration script (`experiment.py`), and shared utilities (`src/`).

## 3. Experiment Directory Structure

Each experiment should be located under the global `experiments/` directory and adhere to the following internal structure:

```plaintext
experiments/
  └── YourExperimentName/               # Root directory for a specific experiment
      ├── experiment.py                 # Main script to define and run the experiment
      ├── config.py                     # Dataclass for experiment-specific configurations
      ├── data_loader.py                # Logic for selecting basin/gauge IDs for this experiment
      ├── yaml_files/                   # Model-specific base hyperparameter configurations
      │   ├── tide.yaml
      │   ├── tsmixer.yaml
      │   ├── ealstm.yaml
      │   └── tft.yaml                  # Add other model YAMLs as needed
      ├── utils.py                      # (Optional) Experiment-specific helper functions
      └── README.md                     # Documentation for this specific experiment
```

## 4. Standard Components

### 4.1. `data_loader.py` - Basin/Gauge ID Selection

* **Purpose**: To define *which* specific basins or gauge IDs will be used in this particular experiment. The method of selection is entirely custom to the experiment.
* **Implementation**:
  * Must implement a function with the signature: `load_data(config: ExperimentConfig, **cli_args: Any) -> List[str]`
    * `config`: An instance of the `ExperimentConfig` dataclass defined in the experiment's `config.py`.
    * `**cli_args`: A dictionary containing any command-line arguments passed to `experiment.py`, allowing `load_data` to be influenced by them if necessary.
    * **Returns**: A `List[str]` containing the unique gauge IDs to be processed by the `HydroInMemoryDataModule`.
  * This script focuses *solely* on producing this list of gauge IDs.

### 4.2. `config.py` - Experiment-Specific Configuration

* **Purpose**: To define all parameters required for the experiment's setup and execution, except for the raw list of gauge IDs (which is dynamically provided by `data_loader.py`).
* **Implementation**:
  * Define a Python dataclass, typically named `ExperimentConfig`.
  * **Structure**:
    * **Tier 1: General & Overrideable Settings**:
      * Parameters for initializing the `HydroInMemoryDataModule`:
        * `region_time_series_base_dirs: Dict[str, Union[str, Path]]`
        * `region_static_attributes_base_dirs: Dict[str, Union[str, Path]]`
        * `path_to_preprocessing_output_directory: Union[str, Path]`
        * `group_identifier: str` (e.g., "gauge\_id")
      * Other datamodule and training parameters:
        * `min_train_years: float`
        * `train_prop: float`, `val_prop: float`, `test_prop: float`
        * `max_imputation_gap_size: int`
        * `chunk_size: int`, `validation_chunk_size: Optional[int]`
        * `num_workers: int`, `batch_size: int`
        * `forcing_features: List[str]`, `static_features: List[str]`, `target: str`
        * `is_autoregressive: bool`
      * `output_dir: str`: Base path for saving all outputs (checkpoints, logs, info files) generated by *this* experiment's execution.
      * (Optional) `default_checkpoint_load_base_dir: Optional[str] = None`: A default base path to look for checkpoints if loading pre-trained models (can be overridden by CLI).
      * These Tier 1 parameters **can be overridden** at runtime via command-line arguments parsed by `experiment.py`.
    * **Tier 2: Data-Loading Specific & Fixed Settings**:
      * Parameters that `data_loader.py` specifically needs to generate its list of gauge IDs (e.g., paths to specific basin list files, attribute thresholds for filtering used within `load_data`).
      * These parameters are considered fixed for the experiment as defined by this `config.py` and are **not typically intended** to be overridden by command-line arguments.
  * The `ExperimentConfig` class may include a `validate()` method for checking inter-parameter consistency and helper methods for constructing derived paths.

### 4.3. `yaml_files/` - Base Model Hyperparameters

* **Purpose**: To store base hyperparameter configurations for each supported model architecture. These define the foundational settings of a model.
* **Structure**: One YAML file per model type (e.g., `tide.yaml`, `tsmixer.yaml`).
* **Content**: Key-value pairs defining model-specific hyperparameters relevant to its architecture and training defaults (e.g., `input_len`, `output_len`, `hidden_size`, `num_layers`, `learning_rate`, `dropout`).
* These files are loaded by `src/models/model_factory.py`. Specific parameters within can be overridden by command-line arguments if the `experiment.py` script is designed to pass them through to the model factory.

### 4.4. `experiment.py` - Main Experiment Orchestrator

* **Purpose**: To parse command-line arguments, set up the experiment environment, orchestrate the training of one or more models over multiple runs and attempts, and manage the generation of standardized outputs.
* **Key Responsibilities**:
    1. **Argument Parsing (`argparse`)**:
        * Arguments to override Tier 1 settings in `ExperimentConfig` (e.g., `--batch-size`, `--output-dir`).
        * `--model-types <type1> [<type2> ...]`: A list of model architectures (e.g., "tide", "tsmixer") to process sequentially.
        * `--yaml-dir <path>`: Path to the directory containing model hyperparameter YAML files (defaults to `./yaml_files/`).
        * `--num-runs <N>`: Number of independent training iterations (logical runs, typically with different seeds) for each specified `model_type`.
        * `--load-checkpoint-from-dir <path>`: (Optional) Path to the base `checkpoints` directory of a *previous* experiment if loading pre-trained models.
        * `--select-best-checkpoint` (boolean flag, default `True`): If loading from `--load-checkpoint-from-dir`, this flag indicates to use the `overall_best_model_info.txt` from the source directory to pick the best checkpoint for the given model type. This overrides `--choose-run-checkpoint` and `--use-attempt`.
        * `--choose-run-checkpoint <run_index>` (integer): If loading and `--select-best-checkpoint` is `False`, specifies the original `run_index` to load from the source directory.
        * `--use-attempt <attempt_index>` (integer, optional): If loading a specific run via `--choose-run-checkpoint`, this specifies which execution "attempt" of that `run_index` to load. If not provided, a default is used (e.g., latest attempt or attempt 0, managed by `checkpoint_manager`).
        * `--override-previous-outputs` (boolean flag, default `False`): If `True`, current execution attempts will overwrite previous output files/directories for the same `model_type/run_index/attempt_index`. If `False`, new "attempts" are versioned.
        * (Optional) Experiment-specific arguments that might modify model hyperparameters further (e.g., `--lr-reduction-factor` for fine-tuning).
    2. **Setup**:
        * Initialize the `ExperimentConfig` object, applying any CLI overrides to Tier 1 settings.
        * Set the global random seed for PyTorch, NumPy, etc. This seed will be the base for per-run seeds.
    3. **Main Experiment Loop (Iterate through `--model-types`)**:
        * For each `model_type` specified in `--model-types`:
            * Locate the appropriate YAML file (e.g., using `args.yaml_dir`).
            * Initialize a list to store results (e.g., `{'checkpoint_path': ..., 'val_loss': ..., 'run_index': ..., 'attempt_index': ...}`) for all execution attempts of all runs for this `model_type` *within the current `experiment.py` invocation*.
            * **Inner Loop (Iterate `args.num_runs` times, for `run_idx` from `0` to `N-1`)**:
                * Set a specific seed for this `run_idx` (e.g., `base_seed + run_idx`).
                * **Determine Output Paths for Current Attempt**: Call `src.experiment_utils.checkpoint_manager.determine_output_run_attempt_path()` to get the unique, versioned output directory path for checkpoints (e.g., `<current_output_dir>/checkpoints/<model_type>/run_<run_idx>/attempt_<attempt_idx>/`) and a similar one for logs. This function handles the creation of new "attempt" subdirectories if `args.override_previous_outputs` is `False` and the logical run (`run_idx`) has been executed before in previous `experiment.py` invocations.
                * **Model Instantiation**:
                    * If `args.load_checkpoint_from_dir` is provided:
                        * Call `src.experiment_utils.checkpoint_manager.get_checkpoint_path_to_load()` with all relevant arguments (`args.load_checkpoint_from_dir`, `model_type`, `args.select_best_checkpoint`, `args.choose_run_checkpoint`, `args.use_attempt`) to resolve the exact `.ckpt` file path from the previous experiment's outputs.
                        * Call `src.models.model_factory.load_pretrained_model()` with this `.ckpt` path, the YAML path (for base config structure), and any fine-tuning specific CLI arguments.
                    * Else (training from scratch):
                        * Call `src.models.model_factory.create_model()` with `model_type`, YAML path, and any CLI arguments intended to override specific base hyperparameters.
                * **DataModule Instantiation**:
                    * Call `data_loader.load_data(experiment_config, **vars(args))` to get the `list_of_gauge_ids`.
                    * Instantiate `HydroInMemoryDataModule` using parameters from `experiment_config` (which includes CLI overrides) and the dynamically loaded `list_of_gauge_ids`.
                    * Call `datamodule.prepare_data()` and `datamodule.setup()`.
                * **Training**:
                    * Configure PyTorch Lightning `Trainer` with standard callbacks:
                        * `EarlyStopping`
                        * `ModelCheckpoint`: Crucially, `dirpath` must point to the specific versioned attempt directory (e.g., `.../run_<run_idx>/attempt_<attempt_idx>/`). `filename` should be descriptive. Set `save_top_k=1` to ensure only the best checkpoint for this specific attempt is retained in its directory.
                        * `LearningRateMonitor`
                    * Configure `TensorBoardLogger`, saving logs to a corresponding versioned attempt directory under `logs/`.
                    * Call `trainer.fit(model, datamodule)`.
                    * After fitting, record the path to the saved best checkpoint (from `trainer.checkpoint_callback.best_model_path`) and its validation loss. Add this information to the list of results for the current `model_type`.
            * **Post-Model Processing (after all `num_runs` for the current `model_type` are complete)**:
                * Analyze the collected list of results (checkpoint paths and validation losses) for the current `model_type` *generated during this execution of `experiment.py`*.
                * Identify the single best performing checkpoint among them.
                * Call `src.experiment_utils.checkpoint_manager.update_overall_best_model_info_file()` to write/overwrite the `overall_best_model_info.txt` in the *current experiment's output directory* (`<current_output_dir>/checkpoints/<model_type>/`) with the relative path to this single best checkpoint.
    4. **Final Output Generation**:
        * Ensure only the following outputs are generated by the experiment script:
            * **Model Checkpoints**: Saved by `ModelCheckpoint` within their respective versioned `run_<run_idx>/attempt_<attempt_idx>/` directories. Typically, only the best checkpoint for each attempt (`save_top_k=1`).
            * **Logs**: TensorBoard logs saved in corresponding versioned `logs/.../run_<run_idx>/attempt_<attempt_idx>/` directories.
            * **`overall_best_model_info.txt`**: For each `model_type` processed, a simple text file located at `<current_output_dir>/checkpoints/<model_type>/overall_best_model_info.txt`. It contains a single line: the relative path (from this `<model_type>` directory) to the single best `.ckpt` file found across all runs and attempts *for that model_type within the current execution of `experiment.py`*.

## 5. `src` Directory - Centralized Utilities

### 5.1. `src/models/model_factory.py`

* **Purpose**: To provide standardized functions for creating new model instances or loading pre-trained ones from checkpoints.
* **Key Functions**:
  * `create_model(model_type: str, yaml_path: str, **override_hp: Any) -> Tuple[pl.LightningModule, Dict[str, Any]]`
  * `load_pretrained_model(model_type: str, yaml_path: str, checkpoint_path: str, **kwargs: Any) -> Tuple[pl.LightningModule, Dict[str, Any]]`
    * This function handles loading the state dict and HPs from the checkpoint.
    * It uses `yaml_path` to understand the base `ModelConfig` structure.
    * `**kwargs` can include parameters like `lr_reduction_factor` or `freeze_backbone` for fine-tuning scenarios, which the function should apply to the loaded model.

### 5.2. `src/utils/hp_from_yaml.py` (or similar)

* **Purpose**: To load and validate base hyperparameter sets from YAML files against model configuration class expectations.

### 5.3. `src/experiment_utils/checkpoint_manager.py` (Crucial for Versioning & Loading)

* **Purpose**: To encapsulate all logic related to:
  * Determining and creating versioned output paths for individual execution attempts.
  * Finding and selecting specific checkpoint files from previous (or current) experiment outputs based on various criteria.
  * Managing the `overall_best_model_info.txt` file.
* **Key Functions (Illustrative Signatures)**:
  * `determine_output_run_attempt_path(base_model_output_dir: Path, run_index: int, override_previous: bool) -> Path`:
    * Input: `<current_output_dir>/checkpoints/<model_type>/`
    * Determines the next available `run_<run_index>/attempt_<N>/` path. If `override_previous` is `True`, it might always target/overwrite `attempt_0` (or just the `run_<run_index>` path without an `attempt` subfolder if overriding means wiping all previous attempts for that run).
    * Creates this directory if it doesn't exist.
    * Returns the resolved `Path` object for the current attempt's output.
    * A similar function would exist for log paths: `determine_log_run_attempt_path(...)`.
  * `get_checkpoint_path_to_load(base_checkpoint_load_dir: Path, model_name: str, select_overall_best: bool, specific_run_index: Optional[int], specific_attempt_index: Optional[int]) -> Path`:
    * `base_checkpoint_load_dir`: Path to the *previous* experiment's `checkpoints` directory (e.g., `/path/to/prev_exp/outputs/checkpoints`).
    * Navigates to `<base_checkpoint_load_dir>/<model_name>/`.
    * If `select_overall_best` is `True`: Reads `overall_best_model_info.txt` from this location and returns the resolved absolute path to the therein specified `.ckpt` file.
    * Else if `specific_run_index` is provided:
      * If `specific_attempt_index` is also provided: Targets `<model_name>/run_<specific_run_index>/attempt_<specific_attempt_index>/` and finds the (expected single, best) `.ckpt` file within.
      * If `specific_attempt_index` is *not* provided: Implements a default rule, e.g., find the highest numbered `attempt_Y` subdirectory within `run_<specific_run_index>/` and use the checkpoint from there.
    * Raises an error if the required checkpoint/info file cannot be found.
  * `update_overall_best_model_info_file(model_checkpoints_output_dir: Path, best_checkpoint_relative_path_within_model_dir: str) -> None`:
    * `model_checkpoints_output_dir`: Path to `<current_output_dir>/checkpoints/<model_type>/`.
    * `best_checkpoint_relative_path_within_model_dir`: The relative path from `model_checkpoints_output_dir` to the best `.ckpt` file identified for the current `model_type` during the current `experiment.py` execution.
    * Writes/overwrites `overall_best_model_info.txt` in `model_checkpoints_output_dir` with this single relative path.

## 6. Output Directory Structure Example

Assuming `ExperimentConfig.output_dir` is `./exp_outputs/MyCurrentExperiment/`.
The `experiment.py` script is run, processing `model_type='tide'`.

* **First execution instance** (`override_previous_outputs=False`):
  * Processes `run_idx=0`, this becomes `attempt_0`.
  * Processes `run_idx=1`, this becomes `attempt_0`.

```plaintext
./exp_outputs/MyCurrentExperiment/
  ├── checkpoints/
  │   └── tide/
  │       ├── overall_best_model_info.txt  # e.g., "run_0/attempt_0/tide_run0_att0_epoch12_val0.123.ckpt" (if that was best from this exec)
  │       ├── run_0/
  │       │   └── attempt_0/               # Output for run_idx=0, first attempt
  │       │       └── tide_run0_att0_epoch12_val0.123.ckpt # Best for this attempt
  │       └── run_1/
  │           └── attempt_0/               # Output for run_idx=1, first attempt
  │               └── tide_run1_att0_epoch10_val0.125.ckpt # Best for this attempt
  └── logs/
      └── tide/
          ├── run_0/
          │   └── attempt_0/               # TensorBoard logs for run_idx=0, attempt_0
          │       └── events.out.tfevents...
          └── run_1/
              └── attempt_0/               # TensorBoard logs for run_idx=1, attempt_0
                  └── events.out.tfevents...
```

* **Second execution instance** of `experiment.py` for the same `output_dir` and `model_type='tide'` (`override_previous_outputs=False`). Suppose `run_idx=0` is processed again (now `attempt_1` for `run_0`), but `run_idx=1` is not part of this second execution or its YAML didn't change:

```plaintext
./exp_outputs/MyCurrentExperiment/
  ├── checkpoints/
  │   └── tide/
  │       ├── overall_best_model_info.txt  # Updated if run_0/attempt_1 is better than previous best.
  │       ├── run_0/
  │       │   ├── attempt_0/               # From first execution
  │       │   │   └── tide_run0_att0_epoch12_val0.123.ckpt
  │       │   └── attempt_1/               # Output for run_idx=0, second attempt (new)
  │       │       └── tide_run0_att1_epoch15_val0.111.ckpt
  │       └── run_1/                       # Unchanged from first execution
  │           └── attempt_0/
  │               └── tide_run1_att0_epoch10_val0.125.ckpt
  └── logs/
      └── tide/
          ├── run_0/
          │   ├── attempt_0/
          │   │   └── ...
          │   └── attempt_1/               # New logs for run_idx=0, attempt_1
          │       └── ...
          └── run_1/
              └── attempt_0/
                  └── ...
```

**Key aspects of the output structure:**

* **`overall_best_model_info.txt`**:
  * Located at `<output_dir>/checkpoints/<model_type>/overall_best_model_info.txt`.
  * Contains a single line: the relative path (from the directory it's in) to the best performing `.ckpt` file identified across all runs and their attempts *for that model_type specifically within the `experiment.py` execution that generated/updated this file*.
* **Run & Attempt Directories**: Named `run_<index>/attempt_<index>/`. This structure provides clear versioning.
* **Checkpoint Files**: One best checkpoint per `attempt` directory, named descriptively (e.g., using `ModelCheckpoint` with `save_top_k=1`).
* **Log Directories**: Mirror the checkpoint `run/attempt` directory structure.

## 7. Example Command-Line Interface (CLI) for `experiment.py`

```bash
# Example: Train tide and tsmixer, 3 runs each, save to a timestamped output directory
python experiments/MyExperiment/experiment.py \
    --config-path ./config.py \
    --model-types tide tsmixer \
    --num-runs 3 \
    --output-dir ./outputs/MyExperimentResults_$(date +%Y%m%d_%H%M%S) \
    --batch-size 1024 \
    --override-previous-outputs False # Default, can be omitted

# Example: Fine-tune 'tide' from a previous experiment's best model, specific run 0, attempt 1
python experiments/MyFineTuningExperiment/experiment.py \
    --config-path ./config.py \
    --model-types tide \
    --num-runs 1 \
    --output-dir ./outputs/FineTuning_Tide_$(date +%Y%m%d_%H%M%S) \
    --load-checkpoint-from-dir ../PreviousExperiment/outputs/checkpoints \
    --select-best-checkpoint False \
    --choose-run-checkpoint 0 \
    --use-attempt 1 \
    --lr-reduction-factor 10
