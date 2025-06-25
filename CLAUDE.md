# Hydro-Forecasting Codebase Structure Guide

## Overview

This codebase implements a comprehensive hydrological time series forecasting system supporting multiple deep learning architectures. The project has undergone a major refactoring (completed in two phases) to migrate away from Railway-Oriented Programming (ROP) to idiomatic Python exception handling, resulting in cleaner, more maintainable code.

## Project Tree Structure

```
hydro-forecasting/
├── src/hydro_forecasting/
│   ├── data/                                    # Data processing and loading modules
│   │   ├── preprocessing.py                     # Main data processing orchestration
│   │   ├── in_memory_datamodule.py             # Lightning DataModule for in-memory data
│   │   ├── in_memory_dataset.py                # PyTorch Dataset implementation
│   │   ├── clean_data.py                       # Data cleaning utilities
│   │   ├── datamodule_validators.py            # Validation logic for data modules
│   │   ├── caravanify_parquet.py               # Basin/gauge data loading utilities
│   │   └── data_deprecated/                    # Legacy data modules (includes lazy_datamodule.py)
│   ├── models/                                  # Model architectures and configurations
│   │   ├── base/                                # Base classes and configurations
│   │   │   └── base_config.py                  # Base configuration class for all models
│   │   ├── tide/                                # TiDE (Time-series Dense Encoder) model
│   │   │   ├── model.py                        # TiDE model implementation
│   │   │   └── config.py                       # TiDE-specific configuration
│   │   ├── tsmixer/                             # TSMixer model architecture
│   │   │   ├── model.py                        # TSMixer model with MLP-based mixing
│   │   │   └── config.py                       # TSMixer-specific configuration
│   │   ├── ealstm/                              # Entity-Aware LSTM model
│   │   │   ├── model.py                        # EA-LSTM implementation
│   │   │   └── config.py                       # EA-LSTM configuration
│   │   ├── tft/                                 # Temporal Fusion Transformer
│   │   │   ├── model.py                        # TFT model implementation
│   │   │   └── config.py                       # TFT configuration
│   │   ├── layers/                              # Shared neural network layers
│   │   │   ├── __init__.py                     # Exports RevIN normalization layer
│   │   │   └── rev_in.py                       # Reversible Instance Normalization
│   │   └── model_factory.py                    # Central model registry and factory
│   ├── preprocessing/                           # Data preprocessing pipeline
│   │   ├── base.py                             # Base transformer classes
│   │   ├── grouped.py                          # Grouped preprocessing pipelines
│   │   ├── unified.py                          # Unified preprocessing pipeline
│   │   ├── pipeline_builder.py                # Builder pattern for preprocessing configs
│   │   ├── transformer_registry.py            # Registry for preprocessing transformers
│   │   ├── log_scale.py                        # Log scaling transformer
│   │   ├── normalize.py                        # Normalization transformer
│   │   └── standard_scale.py                   # Standard scaling transformer
│   ├── model_evaluation/                       # Model evaluation and metrics
│   │   ├── evaluators.py                       # TSForecastEvaluator class (robust, minimal changes needed)
│   │   ├── hp_from_yaml.py                     # YAML configuration loading utilities
│   │   └── checkpoint_manager.py               # Checkpoint discovery and management (critical infrastructure)
│   ├── experiment_utils/                       # Experiment orchestration utilities
│   │   ├── training_runner.py                  # Training execution and management
│   │   └── checkpoint_manager.py               # Versioned checkpoint and output management
│   ├── exceptions.py                           # Custom exception classes (part of ROP migration)
│   └── config_utils.py                         # Configuration utilities and helpers
├── experiments/                                 # Self-contained experiment directories
│   └── [ExperimentName]/                       # Each experiment is self-contained
│       ├── experiment.py                       # Main experiment orchestration script
│       ├── config.py                           # Experiment-specific configuration dataclass
│       ├── data_loader.py                      # Basin/gauge ID selection logic
│       ├── yaml_files/                         # Model hyperparameter configurations
│       │   ├── tide.yaml                       # TiDE model hyperparameters
│       │   ├── tsmixer.yaml                    # TSMixer model hyperparameters
│       │   ├── ealstm.yaml                     # EA-LSTM model hyperparameters
│       │   └── tft.yaml                        # TFT model hyperparameters
│       ├── utils.py                            # Experiment-specific helper functions
│       └── README.md                           # Experiment documentation
├── tests/                                       # Comprehensive test suite
│   └── unit/                                   # Unit tests
│       └── test_preprocessing.py               # Preprocessing pipeline tests
└── docs/                                       # Project documentation
    ├── data_processing.md                      # Data processing pipeline documentation
    ├── experiment-setup-guidelines.md         # Standardized experiment setup guide
    ├── migrate_away_from_ROP_report.md        # ROP migration project summary
    ├── refactoring_vision_plan.md             # Future refactoring plans
    └── streamline_evaluation_plan.md          # Evaluation workflow streamlining plan
```

## Core Architecture Components

### 1. Data Processing Pipeline (`src/hydro_forecasting/data/`)

**Key Module**: `preprocessing.py` - Main orchestration of the data processing pipeline

**Key Features**:

- Loads raw time series data from multiple basins/catchments
- Merges static basin attributes with time series data
- Applies configurable preprocessing transformations
- Performs quality checking and filtering
- Splits data into training/validation/test sets
- Generates deterministic UUIDs for reproducible preprocessing runs

**Data Module**: `in_memory_datamodule.py` - Lightning DataModule for efficient data loading

- Handles configuration validation and reuse of existing processed data
- Creates train/val/test DataLoaders with proper batching
- Supports caching with `_SUCCESS` markers for complete processing runs

**Output Structure**:

```
<preprocessing_output_dir>/
└── <run_uuid>/
    ├── config.json                             # Processing configuration
    ├── pipelines.joblib                        # Fitted preprocessing pipelines
    ├── quality_report.json                     # Data quality metrics
    ├── processed_timeseries/                   # Processed time series data
    │   ├── train/                              # Training split
    │   ├── val/                                # Validation split
    │   └── test/                               # Test split
    ├── processed_static_attributes.parquet     # Processed static features
    └── _SUCCESS                                # Completion marker
```

### 2. Model Architectures (`src/hydro_forecasting/models/`)

**Supported Models**:

- **TiDE**: Time-series Dense Encoder for efficient temporal modeling
- **TSMixer**: MLP-based architecture with separate temporal and feature mixing
- **EA-LSTM**: Entity-Aware LSTM with attention mechanisms
- **TFT**: Temporal Fusion Transformer with interpretability features

**Key Module**: `model_factory.py` - Central model registry and factory

```python
MODEL_REGISTRY = {
    "tide": {"config_class_getter": _get_tide_config, "model_class_getter": _get_tide_model},
    "tsmixer": {"config_class_getter": _get_tsmixer_config, "model_class_getter": _get_tsmixer_model},
    "ealstm": {"config_class_getter": _get_ealstm_config, "model_class_getter": _get_ealstm_model},
    "tft": {"config_class_getter": _get_tft_config, "model_class_getter": _get_tft_model},
}
```

**Configuration System**: Each model has a dedicated config class inheriting from `BaseConfig`

- Separates `STANDARD_PARAMS` (common across models) from `MODEL_PARAMS` (model-specific)
- Supports dynamic loading from YAML files via `hp_from_yaml()`

### 3. Preprocessing System (`src/hydro_forecasting/preprocessing/`)

**Architecture**: Transformer-based preprocessing with registry pattern

- `HydroTransformer`: Base class for all preprocessing transformers
- `GroupedPipeline`: Handles group-aware transformations (e.g., per-basin normalization)
- `PipelineBuilder`: Builder pattern for constructing preprocessing configurations

**Available Transformers**:

- `StandardScaleTransformer`: Z-score normalization
- `NormalizeTransformer`: Min-max scaling
- `LogTransformer`: Logarithmic transformation

**Usage Pattern**:

```python
builder = PipelineBuilder()
config = (builder
    .features().add("standard_scale")
    .target().add("log_transform").add("standard_scale")
    .build())
```

### 4. Experiment Management (`experiments/` and `src/hydro_forecasting/experiment_utils/`)

**Experiment Structure**: Self-contained experiments with standardized layout

- `experiment.py`: Main orchestration script with CLI argument parsing
- `config.py`: Dataclass for experiment-specific configurations
- `yaml_files/`: Model hyperparameter configurations

**Versioned Output Structure**:

```
<experiment_output>/
├── checkpoints/
│   └── <model_type>/
│       ├── overall_best_model_info.txt         # Best checkpoint reference
│       ├── run_0/
│       │   ├── attempt_0/                      # First execution
│       │   └── attempt_1/                      # Re-run with same config
│       └── run_1/
│           └── attempt_0/
└── logs/                                       # Mirror structure for logs
```

**Key Features**:

- Deterministic run versioning with `run_<index>/attempt_<index>/` structure
- Checkpoint reuse and fine-tuning support
- Robust error handling and logging

### 5. Model Evaluation (`src/hydro_forecasting/model_evaluation/`)

**Key Module**: `evaluators.py` - Contains `TSForecastEvaluator` class

- Robust caching mechanism for evaluation results
- Multi-horizon forecasting evaluation
- Comprehensive error handling and logging

**Current State**: Well-designed, requires minimal changes
**Future Enhancement**: Streamlining from ~200 lines of setup code to 5-10 lines via factory pattern

**Checkpoint Management**: `checkpoint_manager.py` - Critical infrastructure

- `get_checkpoint_path_to_load()`: Flexible checkpoint discovery
- Support for "overall best" vs. specific run/attempt selection
- Production-ready with comprehensive error handling

## Key Design Patterns and Principles

### 1. Configuration-Driven Design

- YAML files for model hyperparameters
- Dataclass configurations with validation
- Environment-agnostic path resolution

### 2. Factory Pattern

- `model_factory.py` for model instantiation
- Registry-based transformer system
- Dynamic imports for loose coupling

### 3. Exception-Based Error Handling

- **Major Refactoring Completed**: Migrated away from Railway-Oriented Programming
- Custom exception hierarchy in `exceptions.py`:
  - `ConfigurationError`: Configuration validation failures
  - `FileOperationError`: File I/O related errors  
  - `DataProcessingError`: Data processing failures

### 4. Reproducibility and Versioning

- Deterministic UUIDs for preprocessing runs
- Versioned experiment outputs with attempt tracking
- Comprehensive configuration logging

## Critical Files for LLM Context

### Most Important for Understanding Core Functionality

1. **`src/hydro_forecasting/models/model_factory.py`** - Central model registry, critical for model instantiation
2. **`src/hydro_forecasting/experiment_utils/checkpoint_manager.py`** - Checkpoint discovery and versioning logic
3. **`src/hydro_forecasting/data/in_memory_datamodule.py`** - Main data loading interface
4. **`src/hydro_forecasting/preprocessing/pipeline_builder.py`** - Preprocessing configuration system

### Model-Specific Files

- `src/hydro_forecasting/models/{tide,tsmixer,ealstm,tft}/config.py` - Model configurations
- `src/hydro_forecasting/models/{tide,tsmixer,ealstm,tft}/model.py` - Model implementations

### Evaluation and Infrastructure

- **`src/hydro_forecasting/model_evaluation/evaluators.py`** - Robust evaluation framework
- **`src/hydro_forecasting/model_evaluation/hp_from_yaml.py`** - Configuration loading utilities

## Current Development Status

### Recently Completed

- **Phase 1 & 2 ROP Migration**: Complete refactoring away from `returns` library to standard Python exceptions
- **Data Processing Pipeline**: Robust, production-ready with comprehensive testing
- **Model Factory System**: Dynamic model registry with proven checkpoint loading

### Active Development

- **Evaluation Workflow Streamlining**: Reducing evaluation setup from ~200 lines to 5-10 lines
- **Factory-Based Configuration**: Automatic discovery and configuration of model-experiment combinations
- **Multi-Country Support**: Batch processing capabilities across different regions

### Architecture Strengths

1. **Modular Design**: Clear separation between data, models, preprocessing, and evaluation
2. **Configuration-Driven**: Minimal code changes needed for new models/experiments
3. **Robust Error Handling**: Comprehensive exception hierarchy with detailed error messages
4. **Reproducibility**: Deterministic processing with versioned outputs
5. **Testing Coverage**: Comprehensive unit tests for critical components

## Usage Patterns for LLMs

### When Adding New Models

1. Create model implementation in `src/hydro_forecasting/models/{model_name}/`
2. Add to `MODEL_REGISTRY` in `model_factory.py`
3. Create YAML configuration file for experiments
4. Model automatically integrates with existing evaluation infrastructure

### When Working with Data

1. Use `PipelineBuilder` for preprocessing configuration
2. Leverage existing `HydroInMemoryDataModule` for data loading
3. Follow deterministic UUID pattern for reproducible preprocessing

### When Running Experiments

1. Create self-contained experiment directory under `experiments/`
2. Use standardized `experiment.py` pattern with CLI arguments
3. Leverage existing checkpoint management for versioning and reuse

## Development Environment and Tooling

### Package Management with `uv`

This project uses **uv**, an extremely fast Python package and project manager written in Rust that serves as a single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more. UV is 10-100 times faster than traditional package managers like pip, making dependency installation and environment management significantly more efficient.

**Key Features of uv in this project**:

- **Unified Dependency Management**: Single `pyproject.toml` configuration for all dependencies
- **Fast Virtual Environment Creation**: ~80x faster than `python -m venv`  
- **Automatic Python Installation**: Can install and manage Python versions automatically
- **Project Initialization**: Scaffolds complete Python projects with proper structure

**Essential uv Commands for LLMs**:

```bash
# Install dependencies and sync environment
uv sync

# Add a new dependency 
uv add package-name

# Add a development dependency
uv add --dev package-name

# Run commands in the project environment
uv run python script.py
uv run pytest

# Create a virtual environment (if needed manually)
uv venv

# Install Python versions
uv python install 3.10 3.11 3.12

# Run scripts defined in pyproject.toml
uv run hydro-forecasting          # Main application
uv run hii                        # Human influence index script
uv run cluster_all                # Basin clustering script
```

### Code Quality with `ruff`

This project uses **ruff**, an extremely fast Python linter and code formatter written in Rust that is 10-100x faster than existing linters (like Flake8) and formatters (like Black). Ruff provides over 800 built-in rules with native re-implementations of popular Flake8 plugins.

**Ruff Configuration** (from `pyproject.toml`):

```toml
[tool.ruff]
line-length = 120                    # Maximum line length
target-version = "py310"             # Target Python 3.10+
fix = false                          # Don't auto-fix by default

[tool.ruff.lint]
# Selected rule sets: Pyflakes (E/F), pycodestyle (W), pep8-naming (N), 
# isort (I), pyupgrade (UP), flake8-bugbear (B), flake8-comprehensions (C4), 
# flake8-simplify (SIM)
select = ["E", "F", "W", "N", "I", "UP", "B", "C4", "SIM"]
```

**Essential ruff Commands for LLMs**:

```bash
# Check code for linting errors
uv run ruff check .

# Check specific files
uv run ruff check src/hydro_forecasting/

# Format code
uv run ruff format .

# Auto-fix linting errors where possible
uv run ruff check --fix .

# Check formatting without making changes
uv run ruff format --check .
```

### Project Dependencies

The project uses a modern Python data science and machine learning stack:

**Core Dependencies**:

- **PyTorch (`torch>=2.7.0`)**: Deep learning framework
- **Lightning (`lightning>=2.5.1.post0`)**: PyTorch Lightning for training orchestration
- **Polars (`polars>=1.29.0`)**: Fast DataFrame library (primary data processing)
- **Pandas (`pandas>=2.2.3`)**: Traditional DataFrame operations (legacy support)
- **scikit-learn (`scikit-learn>=1.6.1`)**: Machine learning utilities and preprocessing
- **NumPy (`numpy>=2.2.5`)**: Numerical computing foundation

**Geospatial and Visualization**:

- **CartoPy (`cartopy>=0.24.1`)**: Geospatial data processing and cartographic projections
- **GeoPandas (`geopandas>=1.0.1`)**: Geospatial data analysis
- **Matplotlib (`matplotlib>=3.10.1`)**: Plotting and visualization
- **Seaborn (`seaborn>=0.13.2`)**: Statistical data visualization

**Experiment Management**:

- **Optuna (`optuna>=4.3.0`)**: Hyperparameter optimization
- **Hydra (`hydra-core>=1.3.2`)**: Configuration management
- **TensorBoard (`tensorboard>=2.19.0`)**: Experiment tracking and visualization

**Development Tools**:

- **pytest (`pytest>=8.3.5`)**: Testing framework  
- **ruff (`ruff>=0.11.8`)**: Linting and formatting
- **psutil (`psutil>=7.0.0`)**: System monitoring
- **pympler (`pympler>=1.1`)**: Memory profiling

### Development Workflow for LLMs

When working with this codebase:

1. **Environment Setup**:

   ```bash
   # Clone repository and navigate to project
   cd hydro-forecasting
   
   # Sync dependencies (creates .venv automatically)
   uv sync
   ```

2. **Running Code**:

   ```bash
   # Run any Python script in the project environment
   uv run python src/hydro_forecasting/script.py
   
   # Run tests
   uv run pytest
   
   # Execute project scripts
   uv run hydro-forecasting
   ```

3. **Code Quality Checks**:

   ```bash
   # Format code before committing
   uv run ruff format .
   
   # Check for linting issues
   uv run ruff check .
   
   # Auto-fix issues where possible
   uv run ruff check --fix .
   ```

4. **Adding Dependencies**:

   ```bash
   # Add runtime dependency
   uv add numpy
   
   # Add development dependency  
   uv add --dev pytest-cov
   ```

### Important Notes for LLMs

- **Always use `uv run`**: Prefix Python commands with `uv run` to ensure they execute in the correct environment
- **No manual virtualenv activation needed**: uv handles virtual environment management automatically
- **Fast feedback loops**: Ruff's speed enables real-time code quality checking
- **Modern Python features**: Target Python 3.10+ as specified in the configuration
- **Line length**: Maintain 120 character line limit as configured in ruff settings

This codebase represents a mature, well-architected system with clear patterns and robust infrastructure for hydrological forecasting research and development.
