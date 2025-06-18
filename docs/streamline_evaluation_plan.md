# Comprehensive Refactoring Plan: Hydro Forecasting Evaluation Workflow

## [CONTEXT]

**Current Situation:** The hydro forecasting evaluation workflow requires ~200 lines of repetitive boilerplate code to evaluate multiple model architectures (TFT, EA-LSTM, TiDE, TSMixer) across different training strategies (benchmark, pretrained, finetuned). This includes manual configuration of data modules, checkpoint discovery, model instantiation, and pipeline setup for each combination. The code is duplicated across different countries/regions and becomes exponentially complex when adding new models or experiment types.

**End Goal:** Create a streamlined, configuration-driven evaluation system that reduces the workflow from ~200 lines to 5-10 lines while maintaining full functionality, flexibility, and backwards compatibility. Enable easy addition of new models, countries, and experiment types through simple configuration changes.

## [RELEVANT CODEBASE FILES]

- `hydro_forecasting/model_evaluation/evaluators.py` - Main TSForecastEvaluator class (robust, needs minimal changes)
- `hydro_forecasting/model_evaluation/model_factory.py` - **CRITICAL**: Existing model creation infrastructure
- `hydro_forecasting/model_evaluation/checkpoint_manager.py` - **CRITICAL**: Existing checkpoint discovery infrastructure
- `hydro_forecasting/data/preprocessing.py` - Data preprocessing pipeline orchestration
- `hydro_forecasting/data/in_memory_datamodule.py` - HydroInMemoryDataModule for data loading
- `hydro_forecasting/model_evaluation/hp_from_yaml.py` - YAML configuration loading utilities
- `hydro_forecasting/models/*/` - Model implementations (tft, ealstm, tide, tsmixer)
- `hydro_forecasting/models/base/base_config.py` - Base configuration class
- `hydro_forecasting/data/caravanify_parquet.py` - Basin loading utilities

---

## 1. Executive Summary & Vision

### 1.1. Core Problem

The current evaluation workflow requires manual, repetitive configuration of identical data modules, model loading, and pipeline setup for each model-experiment-country combination, resulting in ~200 lines of boilerplate code that scales poorly and is error-prone to maintain.

### 1.2. Proposed Solution

Create a configuration-driven factory system that leverages the existing `model_factory.py` and `checkpoint_manager.py` infrastructure to automatically discover, load, and configure all model-experiment combinations based on simple declarative configuration. The new system will use path templates for automatic resource discovery, intelligent defaults for common parameters, and graceful error handling for missing components.

### 1.3. Vision Snippet (End-User Code)

```python
# Replace ~200 lines with this:
config = HydroEvaluationConfig(
    country="kyrgyzstan",
    model_types=["tft", "ealstm", "tide", "tsmixer"],
    experiment_types=["benchmark", "pretrained", "finetuned"],
    base_paths={
        "experiments": "/path/to/experiments",
        "data": "/path/to/CaravanifyParquet", 
        "cache": "/path/to/cache"
    }
)

evaluator = create_hydro_evaluator(config)
results = evaluator.test_models(start_of_season=4, end_of_season=10)
```

### 1.4. Backend/System Flow

When `create_hydro_evaluator(config)` is called: (1) Path templates resolve YAML and checkpoint locations, (2) Model factory creates config objects from YAMLs, (3) Checkpoint manager discovers available checkpoints, (4) Model factory loads models from checkpoints, (5) Single data module template creates type-specific instances, (6) Components are assembled into the `models_and_datamodules` dict, (7) TSForecastEvaluator is instantiated with the assembled components.

---

## 2. Current State Analysis

### 2.1. Key Files & Responsibilities

**evaluators.py**: Contains `TSForecastEvaluator` class with robust caching, error handling, and evaluation logic. This is well-designed and requires minimal changes - only potential auto-cache path generation.

**model_factory.py**: **CRITICAL EXISTING INFRASTRUCTURE** - Contains `MODEL_REGISTRY`, `create_model()`, `load_pretrained_model()`, and dynamic import handling. This eliminates the need to create new model registries and provides proven model instantiation patterns with learning rate adjustment for fine-tuning.

**checkpoint_manager.py**: **CRITICAL EXISTING INFRASTRUCTURE** - Contains `get_checkpoint_path_to_load()` with robust error handling, flexible selection (overall best vs specific runs), and production-ready checkpoint discovery. Eliminates need to implement complex file system logic.

**in_memory_datamodule.py**: `HydroInMemoryDataModule` class handles data loading, preprocessing, and batching. Currently requires manual instantiation for each model type despite nearly identical parameters.

**preprocessing.py**: Contains `run_hydro_processor()` and preprocessing pipeline orchestration. The preprocessing logic is solid but configuration setup is repetitive.

**hp_from_yaml.py**: Provides `hp_from_yaml()` function for loading model hyperparameters. This works well and will be leveraged for configuration loading.

**caravanify_parquet.py**: Contains basin loading logic that's currently embedded in evaluation scripts but should be extracted into reusable functions.

### 2.2. Code to Leverage

- **model_factory.py**: Complete module - `MODEL_REGISTRY`, `create_model()`, `load_pretrained_model()`, error handling, learning rate adjustment
- **checkpoint_manager.py**: Complete module - `get_checkpoint_path_to_load()`, robust error handling, path resolution logic
- **hp_from_yaml()**: YAML loading functionality  
- **TSForecastEvaluator**: Complete evaluation logic and caching system
- **HydroInMemoryDataModule**: Data loading infrastructure (need template for instantiation)
- **preprocessing pipeline**: GroupedPipeline and transformer setup patterns

---

## 3. Proposed Architecture

### 3.1. Architectural Diagram/Flow

```
HydroEvaluationConfig
        │
        ▼
create_hydro_evaluator()
        │
        ├─► PathResolver ──► resolve_yaml_paths()
        │                  ├─► resolve_checkpoint_paths()  
        │                  └─► resolve_cache_path()
        │
        ├─► ConfigurationManager ──► load_model_configs() [via hp_from_yaml]
        │                          └─► create_preprocessing_config()
        │
        ├─► CheckpointDiscovery ──► get_checkpoint_path_to_load() [EXISTING]
        │                       └─► graceful error handling [EXISTING]
        │
        ├─► ModelOrchestrator ──► create_model_instances() [via model_factory]
        │                     └─► create_data_modules()
        │
        └─► AssemblyManager ──► build_models_and_datamodules_dict()
                              └─► create_evaluator_instance()
                                      │
                                      ▼
                              TSForecastEvaluator
```

### 3.2. New Components & Files

**hydro_forecasting/evaluation/config.py**: Configuration management with `HydroEvaluationConfig` dataclass, default values, and validation logic.

**hydro_forecasting/evaluation/evaluation_factory.py**: Main orchestration module with `create_hydro_evaluator()` and supporting factory functions that leverage existing infrastructure.

**hydro_forecasting/evaluation/path_resolver.py**: Path template system for automatic resource discovery and resolution.

**hydro_forecasting/evaluation/defaults.py**: Default feature lists, preprocessing configurations, and other constants to reduce boilerplate.

### 3.3. High-Level Snippets (Architecture)

**config.py**:

```python
@dataclass
class HydroEvaluationConfig:
    # Core parameters
    country: str
    model_types: list[str]
    experiment_types: list[str]
    
    # Optional overrides with smart defaults
    forcing_features: list[str] = None  # Uses DEFAULT_FORCING_FEATURES if None
    static_features: list[str] = None   # Uses DEFAULT_STATIC_FEATURES if None
    target: str = "streamflow"
    
    # Path configuration
    base_paths: dict[str, str] = None
    
    # Data module parameters (sensible defaults)
    batch_size: int = 2048
    num_workers: int = 4
    chunk_size: int = 100
    
    # Evaluation parameters  
    horizons: list[int] = None  # Defaults to range(1, 11)
    include_dummy: bool = True
    
    def __post_init__(self):
        # Set defaults and validate
        self._set_defaults()
        self._validate()
```

**evaluation_factory.py**:

```python
from ..model_evaluation.model_factory import create_model, load_pretrained_model
from ..model_evaluation.checkpoint_manager import get_checkpoint_path_to_load

def create_hydro_evaluator(config: HydroEvaluationConfig) -> TSForecastEvaluator:
    """Main factory function - orchestrates entire setup process"""

def load_model_configs(config: HydroEvaluationConfig) -> dict[str, dict]:
    """Load YAML configs for all model types using hp_from_yaml"""
    
def discover_all_checkpoints(config: HydroEvaluationConfig) -> dict[str, dict[str, Path]]:
    """Discover checkpoints using existing checkpoint_manager infrastructure"""
    path_resolver = PathResolver(config.base_paths)
    checkpoints = {}
    
    for model_type in config.model_types:
        checkpoints[model_type] = {}
        
        for experiment_type in config.experiment_types:
            checkpoint_dir = path_resolver.resolve_checkpoint_dir(experiment_type, config.country)
            
            # Use existing checkpoint_manager function
            checkpoint_result = get_checkpoint_path_to_load(
                base_checkpoint_load_dir=Path(checkpoint_dir),
                model_type=model_type,
                select_overall_best=True
            )
            
            if isinstance(checkpoint_result, Success):
                checkpoints[model_type][experiment_type] = checkpoint_result.unwrap()
            else:
                logger.warning(f"No checkpoint found for {model_type}_{experiment_type}")
    
    return checkpoints
    
def create_model_instances(config: HydroEvaluationConfig, checkpoints: dict) -> dict[str, Any]:
    """Create model instances using model_factory functions"""
    
def create_data_modules(config: HydroEvaluationConfig, model_configs: dict) -> dict[str, HydroInMemoryDataModule]:
    """Create data module instances for each model type"""
    
def get_basin_ids_for_country(country: str, base_paths: dict) -> list[str]:
    """Extract and reuse basin loading logic"""
```

**path_resolver.py**:

```python
class PathResolver:
    """Template-based path resolution system"""
    
    def __init__(self, base_paths: dict[str, str]):
        self.base_paths = base_paths
        self.templates = {
            "yaml": "{experiments}/yaml-files/{country}/{model_type}.yaml",
            "checkpoints": "{experiments}/{experiment_type}/{experiment_type}_{country}/checkpoints",
            "cache": "{cache}/model_evaluation_cache_{country}"
        }
    
    def resolve_yaml_path(self, model_type: str, country: str) -> str:
    def resolve_checkpoint_dir(self, experiment_type: str, country: str) -> str:
    def resolve_cache_path(self, country: str) -> str:
```

**defaults.py**:

```python
DEFAULT_FORCING_FEATURES = [
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

DEFAULT_STATIC_FEATURES = [
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

def create_default_preprocessing_config():
    """Create standard preprocessing configuration"""
```

---

## 4. Phased Implementation Roadmap

### Phase 1: Configuration Foundation

**Files to Create**: `config.py`, `defaults.py`
**Files to Modify**: None
**Goal**: Establish configuration system and defaults

**Tasks**:

1. Create `HydroEvaluationConfig` dataclass with all required fields
2. Implement `DEFAULT_FORCING_FEATURES` and `DEFAULT_STATIC_FEATURES` constants
3. Add configuration validation in `__post_init__()`
4. Create default preprocessing configuration function

**Acceptance Criteria**: Configuration object can be instantiated with minimal parameters and provides sensible defaults

### Phase 2: Path Resolution System

**Files to Create**: `path_resolver.py`
**Files to Modify**: None
**Goal**: Automatic path discovery and resolution

**Tasks**:

1. Implement `PathResolver` class with template system
2. Create methods for YAML, checkpoint, and cache path resolution
3. Add error handling for missing base paths
4. Validate path resolution with expected directory structures

**Acceptance Criteria**: Given base paths and parameters, system can resolve all required file/directory paths

### Phase 3: Model Configuration Loading

**Files to Create**: Begin `evaluation_factory.py`
**Files to Modify**: None
**Goal**: Load model configs using existing infrastructure

**Tasks**:

1. Implement `load_model_configs()` using `hp_from_yaml()`
2. Create `get_basin_ids_for_country()` by extracting from existing code
3. Implement `create_preprocessing_config()` with standard pipelines
4. Add error handling for missing YAML files

**Acceptance Criteria**: Can load all model configurations for a given country/model type combination

### Phase 4: Checkpoint Discovery

**Files to Create**: Continue `evaluation_factory.py`
**Files to Modify**: None
**Goal**: Leverage existing checkpoint_manager for discovery

**Tasks**:

1. Import and use `get_checkpoint_path_to_load()` from checkpoint_manager
2. Implement `discover_all_checkpoints()` as thin wrapper around existing function  
3. Add path resolution for different experiment type directories
4. Add logging for missing checkpoints (error handling already exists)

**Acceptance Criteria**: System can discover available checkpoints using proven checkpoint_manager logic

### Phase 5: Model Creation Integration

**Files to Create**: Continue `evaluation_factory.py`
**Files to Modify**: None  
**Goal**: Leverage model_factory for model instantiation

**Tasks**:

1. Implement `create_model_instances()` using `load_pretrained_model()` and `create_model()`
2. Add dummy model creation for baseline comparison
3. Handle learning rate adjustments for fine-tuned models (already in model_factory)
4. Add comprehensive error handling and logging

**Acceptance Criteria**: Can create all available model instances from discovered checkpoints

### Phase 6: Data Module Creation

**Files to Create**: Continue `evaluation_factory.py`
**Files to Modify**: None
**Goal**: Template-based data module creation

**Tasks**:

1. Implement `create_data_modules()` using model configs for hyperparameters
2. Create data module template with smart parameter extraction
3. Handle region and basin ID configuration
4. Add preprocessing pipeline setup

**Acceptance Criteria**: Can create properly configured data modules for each model type

### Phase 7: Main Orchestration

**Files to Create**: Complete `evaluation_factory.py`
**Files to Modify**: None
**Goal**: Complete end-to-end workflow

**Tasks**:

1. Implement `create_hydro_evaluator()` main function
2. Create `build_models_and_datamodules_dict()` assembly function
3. Add comprehensive error handling and user feedback
4. Implement auto-cache path generation

**Acceptance Criteria**: End-to-end workflow works from config to evaluator instance

### Phase 8: Integration & Validation

**Files to Create**: Documentation, examples
**Files to Modify**: Potentially `evaluators.py` for auto-cache paths
**Goal**: Ensure system works with real data and provides backwards compatibility

**Tasks**:

1. Validate complete workflow with actual kyrgyzstan data
2. Verify results match manual 200-line setup
3. Document any discovered edge cases
4. Create usage examples and documentation

**Acceptance Criteria**: New system produces identical results to manual setup

### Phase 9: Enhancement & Polish

**Files to Create**: Additional convenience functions
**Files to Modify**: Add convenience methods
**Goal**: Production-ready system with good UX

**Tasks**:

1. Add auto-cache path generation to `TSForecastEvaluator`
2. Create comprehensive documentation and examples
3. Add configuration validation with helpful error messages
4. Performance optimization and logging improvements

**Acceptance Criteria**: System is production-ready with good documentation and user experience

### Phase 10: Multi-Country Support

**Files to Create**: Batch processing utilities
**Files to Modify**: Extend factory functions
**Goal**: Easy batch processing across countries

**Tasks**:

1. Add batch evaluation support for multiple countries
2. Create configuration templates for common scenarios
3. Add parallel processing capabilities
4. Create utilities for comparative analysis across countries

**Acceptance Criteria**: Can easily evaluate multiple countries in single workflow

---

## 5. Risk Assessment & Mitigation

### Overall Risk: MEDIUM (reduced from HIGH due to existing infrastructure)

**Risk Reduction Factors**:

- **model_factory.py**: Eliminates need to implement model creation logic
- **checkpoint_manager.py**: Eliminates need to implement checkpoint discovery
- **Incremental phases**: Each phase builds on the previous without breaking existing code
- **Backwards compatibility**: Existing 200-line workflows continue to work

**Remaining Risks**:

- **Path template complexity**: Mitigated by starting with simple templates and expanding
- **Configuration validation**: Mitigated by comprehensive validation in early phases
- **Data module template**: Mitigated by leveraging existing successful patterns

### Timeline Estimate

**Total Implementation Time**: 1-2 weeks (reduced from 2-3 weeks due to existing infrastructure)

**Critical Path**: Phases 1, 5, 7, 8 ensure core functionality works before adding enhancements

**Phase 4 specifically**: 1-2 days (checkpoint discovery is now a wrapper function rather than custom implementation)

This refactoring leverages substantial existing infrastructure (`model_factory.py` and `checkpoint_manager.py`) to dramatically reduce implementation complexity while maintaining full functionality and backwards compatibility.
