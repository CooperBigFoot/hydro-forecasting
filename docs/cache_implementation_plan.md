## 1. Detailed File Formats & Schemas

### Cache Metadata (`cache_metadata.json`)

```json
{
    "cache_version": "1.0",
    "created_at": "2025-06-17T10:30:00Z",
    "last_updated": "2025-06-17T15:45:00Z",
    "cached_models": {
        "ealstm_benchmark": {
            "cached_at": "2025-06-17T10:30:00Z",
            "horizons": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "start_of_season": 4,
            "end_of_season": 10,
            "num_basins": 125,
            "num_predictions": 15000,
            "file_size_mb": 2.5
        }
    }
}
```

### Predictions Parquet Schema

- Columns: `["horizon", "observed", "predicted", "date", "gauge_id"]`
- Partitioning: None (small files)
- Compression: snappy (good balance of speed/size)

### Metrics JSON Schema

```json
{
    "gauge_id_1": {
        "1": {"mse": 0.123, "mae": 0.456, "rmse": 0.789, "nse": 0.85, ...},
        "2": {"mse": 0.234, "mae": 0.567, "rmse": 0.890, "nse": 0.82, ...},
        ...
    },
    "gauge_id_2": { ... }
}
```

## 2. Parameter Compatibility & Cache Invalidation

### Cache Key Components

A cached model is valid for retrieval if ALL match:

- Model name (exact string match)
- Horizons list (exact match, order-independent)
- `start_of_season` parameter
- `end_of_season` parameter

### Invalidation Scenarios

- **Different horizons**: Cache miss → re-run with new horizons
- **Different seasonal parameters**: Cache miss → re-run
- **Missing model**: Cache miss → run test
- **Corrupted files**: Treat as cache miss → re-run

## 3. Enhanced Error Handling & Recovery

### File System Errors

```python
# Pseudocode scenarios
try:
    load_cache_metadata()
except FileNotFoundError:
    # First time using cache - create directory structure
    create_cache_directory()
except PermissionError:
    # Log error, fallback to no caching
    log.warning("Cannot access cache due to permissions")
    return run_without_cache()
except JSONDecodeError:
    # Corrupted metadata
    log.warning("Cache metadata corrupted, rebuilding")
    backup_and_rebuild_metadata()
```

### Data Validation

```python
def validate_cached_predictions(df: pd.DataFrame, expected_horizons: list) -> bool:
    """Validate loaded predictions match expected structure."""
    required_columns = ["horizon", "observed", "predicted", "date", "gauge_id"]
    if not all(col in df.columns for col in required_columns):
        return False
    if set(df['horizon'].unique()) != set(expected_horizons):
        return False
    return True
```

## 4. Cache Management Operations

### New Public Methods

```python
class TSForecastEvaluator:
    def clear_cache(self, model_names: list[str] | None = None) -> None:
        """Remove specific models from cache, or entire cache if model_names=None."""
        
    def cache_info(self) -> dict:
        """Return information about cached models and disk usage."""
        
    def validate_cache(self) -> dict[str, bool]:
        """Check integrity of all cached models."""
        
    def get_cache_size(self) -> float:
        """Return total cache size in MB."""
```

### Cache Statistics & Monitoring

```python
cache_info_example = {
    "total_models": 12,
    "total_size_mb": 45.2,
    "oldest_entry": "2025-06-15T09:00:00Z",
    "newest_entry": "2025-06-17T15:45:00Z",
    "models": {
        "ealstm_benchmark": {"size_mb": 2.5, "cached_at": "..."},
        # ...
    }
}
```

## 5. Advanced Configuration Options

### Constructor Parameters

```python
def __init__(
    self,
    horizons: list[int],
    models_and_datamodules: dict[str, tuple],
    trainer_kwargs: dict[str, Any],
    save_path: str | Path | None = None,
    cache_path: str | Path | None = None,  # NEW: default cache location
    cache_compression: str = "snappy",      # NEW: parquet compression
    cache_max_age_days: int | None = None, # NEW: auto-cleanup old entries
) -> None:
```

### Method-Level Cache Control

```python
def test_models(
    self, 
    start_of_season: int | None = None, 
    end_of_season: int | None = None,
    cache_path: str | Path | None = None,           # Override constructor cache_path
    force_refresh: bool = False,                    # Force re-run all models
    force_refresh_models: list[str] | None = None, # Force re-run specific models
    cache_predictions: bool = True,                 # Enable/disable predictions caching
    cache_metrics: bool = True,                     # Enable/disable metrics caching
) -> dict[str, dict]:
```

## 6. Logging Strategy

### Log Levels & Messages

```python
# INFO level (always shown)
"Cache found at /path/to/cache (12 models, 45.2 MB)"
"Loading cached results for: model1, model2 (2/5 models)"  
"Testing models: model3, model4, model5 (3/5 models)"
"Cached model3 results (2.1 MB) to /path/to/cache"

# WARNING level
"Cache file corrupted for model1, re-running test"
"Cache parameters mismatch for model2 (horizons), re-running test"

# DEBUG level  
"Loading predictions for model1 from cache/predictions/model1.parquet"
"Validating cached data for model1: 1250 predictions, 25 basins"
"Cache hit for model1 (horizons=[1,2,3,4,5], season=4-10)"
```

## 7. Performance Optimizations

### Memory Management

- Load cached predictions lazily (only when needed)
- Option to return results without loading all data into memory
- Stream processing for very large prediction sets

### I/O Optimizations

- Parallel loading of multiple cached models
- Efficient parquet reading with column selection
- Batch operations for cache updates

## 8. Testing Strategy

### Unit Tests Required

```python
def test_cache_creation():
    """Test cache directory and metadata creation."""
    
def test_cache_hit():
    """Test loading existing cached model."""
    
def test_cache_miss():
    """Test handling missing model in cache."""
    
def test_parameter_mismatch():
    """Test cache invalidation due to parameter changes."""
    
def test_corrupted_cache():
    """Test recovery from corrupted cache files."""
    
def test_partial_cache():
    """Test mixed cached/non-cached model evaluation."""
    
def test_cache_management():
    """Test clear_cache, cache_info, etc."""
```

### Integration Tests

- Full workflow with real model checkpoints
- Cache persistence across multiple evaluator instances
- Performance benchmarks (cache vs no-cache)

## 9. Backwards Compatibility

### Migration Strategy

- Current `test_models()` calls work unchanged (cache_path=None)
- Existing `save_path` functionality preserved
- No breaking changes to return format

### Version Compatibility

- Cache version field in metadata
- Graceful handling of older cache formats
- Clear error messages for incompatible caches

## 10. Documentation Requirements

### Docstring Updates

- Update `test_models()` docstring with caching examples
- Document all new cache-related methods
- Add caching section to class docstring

### Usage Examples

```python
# Basic caching
results = evaluator.test_models(cache_path="./model_cache")

# Force refresh specific models
results = evaluator.test_models(
    cache_path="./model_cache",
    force_refresh_models=["ealstm_benchmark", "tide_finetuned"]
)

# Cache management
evaluator.cache_info()
evaluator.clear_cache(["old_model_1", "old_model_2"])
evaluator.validate_cache()
```

## 11. Implementation Phases

### Phase 1: Core Caching

- Basic cache save/load functionality
- Simple cache hit/miss logic
- File format implementation

### Phase 2: Error Handling

- Corruption recovery
- Parameter validation
- Comprehensive logging

### Phase 3: Cache Management

- Cache info/statistics
- Cleanup operations
- Advanced configuration options

### Phase 4: Optimization & Polish

- Performance improvements
- Memory optimizations
- Documentation completion
