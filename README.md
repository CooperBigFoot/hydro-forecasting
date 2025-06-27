# Hydro-Forecasting: Deep Learning for Hydrological Time Series Prediction

## 🤖 Quick Reference for AI Assistants

This hydrological forecasting system implements multiple deep learning architectures for streamflow prediction. The codebase follows a modular, configuration-driven design with comprehensive error handling and reproducibility features.

**Key Context Files for AI Understanding:**

- `CLAUDE.md` - Comprehensive codebase structure guide
- `docs/model_implementation_guidelines.md` - Model implementation conventions
- `src/hydro_forecasting/models/model_factory.py` - Central model registry
- `experiments/*/experiment.py` - Experiment orchestration patterns

## 🏗️ Project Architecture Overview

### Core System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │───▶│ Model Training  │───▶│  Evaluation     │
│ (preprocessing) │    │(Lightning-based)│    │  (metrics)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Processed Data  │    │   Checkpoints   │    │    Results      │
│(UUID-versioned) │    │(versioned runs) │    │ (cached/saved)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Supported Model Architectures

1. **TiDE** (Time-series Dense Encoder) - Efficient temporal modeling
2. **TSMixer** - MLP-based architecture with temporal/feature mixing
3. **EA-LSTM** - Entity-Aware LSTM
4. **TFT** (Temporal Fusion Transformer) - Interpretable forecasting

## 📁 Essential Directory Structure

```
hydro-forecasting/
├── src/hydro_forecasting/          # Core library code
│   ├── data/                       # Data loading and processing
│   │   ├── preprocessing.py        # Main data pipeline orchestration
│   │   └── in_memory_datamodule.py # Lightning DataModule
│   ├── models/                     # Model implementations
│   │   ├── {tide,tsmixer,ealstm,tft}/  # Individual model packages
│   │   └── model_factory.py        # MODEL_REGISTRY and factory pattern
│   ├── preprocessing/              # Preprocessing transformers
│   │   └── pipeline_builder.py     # Builder pattern for pipelines
│   └── experiment_utils/           # Training and checkpoint management
│       └── checkpoint_manager.py   # Versioned checkpoint handling
├── experiments/                    # Self-contained experiment directories
│   └── {ExperimentName}/          
│       ├── experiment.py           # Main experiment script
│       ├── config.py              # Experiment configuration dataclass
│       └── yaml_files/            # Model hyperparameters
└── tests/                         # Comprehensive test suite
```

## ⚡ Quick Start Commands

### Environment Setup

```bash
# Install dependencies using uv (fast Rust-based package manager)
uv sync

# Run any Python script in the project environment
uv run python script.py

# Run tests
uv run pytest

# Code quality checks
uv run ruff check .
uv run ruff format .
```

### Common Development Tasks

```bash
# Add new dependency
uv add package-name

# Run experiment
uv run python experiments/ExperimentName/experiment.py \
    --model_type tide \
    --preprocessing_mode unified

# Evaluate model
uv run python evaluation_script.py \
    --checkpoint_path experiments/output/checkpoints/tide/run_0/attempt_0/
```

## 🎯 Key Design Patterns

### 1. Factory Pattern for Models

```python
# src/hydro_forecasting/models/model_factory.py
MODEL_REGISTRY = {
    "tide": {"config_class_getter": _get_tide_config, "model_class_getter": _get_tide_model},
    # ... other models
}
```

### 2. Configuration-Driven Design

- YAML files for hyperparameters
- Dataclass configurations with validation
- Environment-agnostic path resolution

### 3. Preprocessing Pipeline Builder

```python
# Example usage
builder = PipelineBuilder()
config = (builder
    .features().add("standard_scale")
    .target().add("log_transform").add("standard_scale")
    .build())
```

### 4. Versioned Outputs

```
experiments/output/
├── checkpoints/
│   └── tide/
│       ├── overall_best_model_info.txt
│       └── run_0/attempt_0/           # Deterministic versioning
└── logs/                              # Mirror structure
```

## 🔄 Data Processing Pipeline

### Input Data Structure

- **Time Series**: Basin-specific hydrological measurements
- **Static Attributes**: Basin characteristics (area, elevation, etc.)
- **Data Formats**: Parquet files with standardized schema

### Processing Flow

1. Load raw data from multiple basins
2. Merge static attributes with time series
3. Apply preprocessing transformations (configurable)
4. Quality checking and filtering
5. Train/val/test splitting
6. Generate UUID for reproducible runs

### Output Structure

```
<preprocessing_output>/<uuid>/
├── config.json                    # Processing configuration
├── pipelines.joblib              # Fitted preprocessing pipelines
├── processed_timeseries/         # Split data
│   ├── train/
│   ├── val/
│   └── test/
└── _SUCCESS                      # Completion marker
```

## 🧪 Testing Infrastructure

### Test Categories

- **Unit Tests**: Component-level testing
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Speed and memory benchmarks

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/unit/test_preprocessing.py

# With coverage
uv run pytest --cov=hydro_forecasting
```

## 📊 Model Evaluation

### Evaluation Features

- Multi-horizon forecasting metrics
- Robust caching mechanism
- Comprehensive error handling
- Support for ensemble evaluations

### Key Metrics

- RMSE, MAE, NSE (Nash-Sutcliffe Efficiency)
- Per-basin and aggregated metrics
- Temporal analysis capabilities

## 💻 Development Guidelines

### Adding New Models

1. Create model package: `src/hydro_forecasting/models/{model_name}/`
2. Implement `model.py` and `config.py`
3. Register in `MODEL_REGISTRY` in `model_factory.py`
4. Create YAML configuration for experiments

### Code Quality Standards

- Line length: 120 characters
- Python 3.10+ features
- Type annotations required
- Comprehensive docstrings
- Follow exception-based error handling (no ROP)

### Git Workflow

- Feature branches from `main`
- Descriptive commit messages
- Run tests before pushing
- Update documentation for new features

## ⚠️ Important Notes for AI Assistants

### When Working with This Codebase

1. **Always check existing patterns** - The codebase has established conventions
2. **Use factory patterns** - Don't instantiate models directly
3. **Configuration over code** - Prefer YAML configs to hardcoded values
4. **Respect versioning** - Use checkpoint manager for model outputs
5. **Error handling** - Use custom exceptions from `exceptions.py`

### Common Pitfalls to Avoid

- Don't use relative imports outside of packages
- Don't hardcode paths - use configuration
- Don't skip validation - use dataclass validators
- Don't ignore existing utilities - check before reimplementing

### Key Files to Reference

- **For models**: `model_factory.py`, model-specific `config.py`
- **For data**: `preprocessing.py`, `in_memory_datamodule.py`
- **For experiments**: Existing experiment patterns in `experiments/`
- **For evaluation**: `evaluators.py`, `checkpoint_manager.py`

## 🌍 Project Context

### Research Focus

- Hydrological forecasting for water resource management
- Multi-country support (Kyrgyzstan, Tajikistan, etc.)
- Transfer learning across different basins
- Climate-aware predictions

### Recent Development

- Completed migration from Railway-Oriented Programming to exceptions
- Streamlining evaluation workflows
- Enhanced multi-country batch processing
- Improved checkpoint management system

## 📚 Additional Documentation

- `CLAUDE.md` - Detailed codebase structure and patterns
- `docs/model_implementation_guidelines.md` - Model implementation guide
- `tests/README.md` - Testing documentation
- `.github/copilot-instructions.md` - Code style guidelines

## 🤝 Contributing

When contributing to this project:

1. Follow existing code patterns and conventions
2. Add tests for new functionality
3. Update documentation as needed
4. Use `uv` for dependency management
5. Run `ruff` for code quality checks

---

*This README is optimized for AI assistants to quickly understand and work with the hydro-forecasting codebase. For detailed implementation guidelines, refer to the documentation in the `docs/` directory.*
