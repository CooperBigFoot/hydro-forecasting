# Hydro Forecasting Preprocessing Pipeline Test Suite

This directory contains a comprehensive test suite for the hydrological forecasting preprocessing pipeline. The tests are designed to ensure robustness, reliability, and performance of the data processing components.

## Test Structure

```
tests/
├── fixtures/
│   └── conftest.py              # Comprehensive test fixtures and synthetic data
├── unit/
│   ├── test_clean_data.py       # Data cleaning module tests
│   ├── test_preprocessing.py    # Main preprocessing orchestration tests
│   ├── test_grouped_pipeline.py # GroupedPipeline implementation tests
│   └── test_unified_pipeline.py # UnifiedPipeline implementation tests
├── integration/
│   └── test_preprocessing_workflow.py  # End-to-end workflow tests
├── performance/
│   ├── test_memory_usage.py     # Memory efficiency tests
│   └── test_processing_speed.py # Performance benchmarks
└── test_data/
    ├── synthetic_basins/        # Generated test data
    └── config_examples/         # Example configurations
```

## Test Categories

### Unit Tests
- **Data Cleaning (`test_clean_data.py`)**: Tests for the `clean_data()` function, gap detection, imputation, quality reporting, and validation logic
- **Preprocessing (`test_preprocessing.py`)**: Tests for pipeline factory functions, data splitting, batch processing, and utility functions
- **GroupedPipeline (`test_grouped_pipeline.py`)**: Tests for per-group pipeline fitting, transformation, multiprocessing, and error handling
- **UnifiedPipeline (`test_unified_pipeline.py`)**: Tests for unified pipeline fitting, transformation, column processing, and inverse transforms

### Integration Tests
- **Preprocessing Workflow (`test_preprocessing_workflow.py`)**: End-to-end tests for the complete `run_hydro_processor()` workflow, including file I/O, batch processing, and configuration validation

### Performance Tests
- **Memory Usage (`test_memory_usage.py`)**: Tests for memory efficiency, leak detection, and optimization with large datasets
- **Processing Speed (`test_processing_speed.py`)**: Benchmarks for processing speed, multiprocessing performance, and scalability

## Key Features Tested

### Data Quality and Validation
- Missing data detection and handling
- Gap imputation with configurable limits
- Training data sufficiency validation
- Data type and schema validation
- Quality report generation and summarization

### Pipeline Functionality
- GroupedPipeline: Separate pipelines per basin/group
- UnifiedPipeline: Single pipeline for all data
- Pipeline factory and configuration management
- Multiprocessing support for GroupedPipeline
- Transform and inverse transform operations

### Workflow Integration
- Complete preprocessing workflow execution
- Batch processing of large datasets
- File I/O operations and directory management
- Configuration validation and error handling
- Static feature processing integration

### Error Handling
- Comprehensive exception hierarchy testing
- Graceful handling of edge cases
- Informative error messages
- Recovery from processing failures

### Performance and Scalability
- Memory usage monitoring and optimization
- Processing speed benchmarks
- Scalability with data size and number of groups
- Multiprocessing performance gains

## Test Data and Fixtures

### Synthetic Data Generation
The test suite includes sophisticated synthetic data generation that creates realistic hydrological time series with:
- Multiple basins with different characteristics
- Seasonal patterns and trends
- Correlated variables (temperature, precipitation, streamflow)
- Realistic gap patterns and missing data
- Outliers and edge cases

### Configurable Test Scenarios
- Clean data scenarios
- Data with various gap patterns
- Insufficient training data
- Missing columns and schema violations
- Different split proportions
- Various pipeline configurations

## Running the Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-mock psutil
```

### Basic Test Execution
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/performance/    # Performance tests only

# Run specific test files
pytest tests/unit/test_clean_data.py
pytest tests/integration/test_preprocessing_workflow.py
```

### Test Options
```bash
# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=hydro_forecasting --cov-report=html

# Run performance tests (may take longer)
pytest tests/performance/ -m slow

# Run specific test by name
pytest tests/unit/test_clean_data.py::TestCleanDataFunction::test_clean_data_valid_input
```

### Parallel Test Execution
```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto
```

## Test Coverage Goals

The test suite aims for >90% code coverage across all preprocessing modules:

- `hydro_forecasting.data.clean_data`
- `hydro_forecasting.data.preprocessing`
- `hydro_forecasting.preprocessing.grouped`
- `hydro_forecasting.preprocessing.unified`

## Performance Benchmarks

### Expected Performance Targets
- UnifiedPipeline fitting: <1 second for standard test data
- GroupedPipeline fitting: <5 seconds for standard test data
- Memory usage: <2GB for large dataset processing
- No memory leaks in repeated operations

### Hardware Considerations
Performance benchmarks are designed to be reasonable across different hardware configurations. Tests may need adjustment for:
- Low-memory environments
- Single-core systems
- Very slow storage systems

## Debugging Test Failures

### Common Issues and Solutions

1. **Memory-related failures**:
   - Check available system memory
   - Reduce test data size if necessary
   - Ensure proper cleanup in test fixtures

2. **Timing-sensitive failures**:
   - Performance tests may fail on slow systems
   - Adjust timeout values if needed
   - Use `pytest.mark.slow` to skip performance tests

3. **File I/O failures**:
   - Ensure write permissions in test directories
   - Check available disk space
   - Verify temporary directory access

4. **Multiprocessing issues**:
   - Some systems may have multiprocessing limitations
   - Tests include fallbacks for single-threaded execution
   - Check system-specific multiprocessing constraints

### Test Data Issues
If synthetic data generation fails:
- Check random seed consistency
- Verify date range generation
- Ensure numerical stability in calculations

## Contributing to Tests

### Adding New Tests
When adding new functionality:
1. Add unit tests for individual functions
2. Add integration tests for workflow changes
3. Update performance tests if needed
4. Include edge cases and error scenarios

### Test Naming Conventions
- Test classes: `TestFunctionName` or `TestFeatureName`
- Test methods: `test_specific_behavior_description`
- Use descriptive names that explain what is being tested

### Mock Usage Guidelines
- Mock external dependencies (file system, network)
- Don't mock the code under test
- Use mocks to test error scenarios
- Keep mocks simple and focused

### Performance Test Guidelines
- Mark slow tests with `@pytest.mark.slow`
- Include performance expectations in assertions
- Consider hardware variability in benchmarks
- Focus on relative performance rather than absolute values

## Maintenance

### Regular Maintenance Tasks
1. Update synthetic data generation as needed
2. Adjust performance benchmarks for new hardware
3. Add tests for new features and bug fixes
4. Review and update test coverage goals

### Monitoring Test Health
- Run full test suite regularly
- Monitor test execution times
- Track test coverage metrics
- Address flaky or unreliable tests promptly

This comprehensive test suite ensures the reliability and performance of the hydro forecasting preprocessing pipeline across various scenarios and use cases.