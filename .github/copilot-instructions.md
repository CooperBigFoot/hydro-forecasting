# Python Code Generation Guidelines

> Generate high-quality, maintainable Python code that follows modern best practices and industry standards.

## Core Standards

Generate Python code that is **production-ready, maintainable, and follows modern Python conventions**. Every piece of code should be clean, well-documented, and robust enough for real-world applications.

**Compatibility & Style:**
- Use Python 3.10+ features and syntax
- Follow PEP 8 style guidelines with 88-character line limits
- Use 4 spaces for indentation consistently
- Write readable, self-documenting code that other developers can easily understand and maintain

## Import Organization

Structure imports to enhance code clarity and maintainability:

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import requests
import pandas as pd

# Local application imports
from my_package.core import MyClass
from my_package.utils import helper_function
```

**Best Practices:**
- Separate import groups with blank lines
- Use absolute imports for clarity (`from my_package.module import function`)
- Avoid wildcard imports (`from module import *`) to prevent namespace pollution
- Use relative imports only when they significantly improve readability within the same package

## Type Annotations

Implement comprehensive type hints to improve code clarity and enable better tooling support:

```python
def process_data(
    items: list[dict[str, int | float]], 
    threshold: float = 0.5
) -> tuple[list[dict[str, int | float]], int]:
    """Process data items above the specified threshold."""
    filtered_items = [item for item in items if item.get('value', 0) > threshold]
    return filtered_items, len(filtered_items)
```

**Type Annotation Strategy:**
- Add type hints to all function signatures (parameters and return types)
- Annotate variables when types aren't immediately clear from context
- Use modern syntax: `list[int]` instead of `List[int]`, `str | None` instead of `Optional[str]`
- Leverage `typing` module for advanced types: `Callable`, `Protocol`, `TypeVar`, `Any`

## Documentation Excellence

Create comprehensive documentation that explains both the "what" and "why" of your code:

```python
def calculate_weighted_average(
    values: list[float], 
    weights: list[float] | None = None
) -> float:
    """Calculate the weighted average of a list of values.
    
    Computes the weighted average where each value is multiplied by its
    corresponding weight. If no weights are provided, treats all values
    as equally weighted.
    
    Args:
        values: List of numeric values to average. Must not be empty.
        weights: Optional list of weights corresponding to each value.
                If None, equal weights are assumed for all values.
    
    Returns:
        The weighted average as a float.
        
    Raises:
        ValueError: If values is empty or if values and weights have
                   different lengths.
        ZeroDivisionError: If all weights sum to zero.
    
    Example:
        >>> calculate_weighted_average([1, 2, 3], [0.5, 0.3, 0.2])
        1.7
    """
```

**Documentation Guidelines:**
- Use Google-style docstrings unless the codebase uses a different established style
- Document all public modules, classes, functions, and methods
- Include purpose, parameters, return values, exceptions, and usage examples
- Add inline comments for complex logic, focusing on "why" rather than "what"

## Robust Error Handling

Implement defensive programming with specific, actionable error handling:

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def read_config_file(file_path: str | Path) -> dict[str, str]:
    """Read configuration from a JSON file with comprehensive error handling."""
    path = Path(file_path)
    
    try:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with path.open('r', encoding='utf-8') as file:
            config = json.load(file)
            
        logger.info(f"Successfully loaded config from {path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Config file missing: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {path}: {e}")
        raise ValueError(f"Malformed configuration file: {e}") from e
    except PermissionError:
        logger.error(f"Permission denied reading config file: {path}")
        raise
```

**Error Handling Principles:**
- Use specific exception types in `except` clauses
- Log errors with context using the `logging` module
- Never suppress errors silently - always log or re-raise appropriately
- Avoid bare `except:` clauses except for top-level error boundaries with proper logging

## Clean Code Architecture

Design modular, maintainable code that follows SOLID principles:

```python
class DataProcessor:
    """Processes data with configurable transformation strategies."""
    
    def __init__(self, config: dict[str, str]) -> None:
        self._config = config
        self._validators: list[Callable[[dict], bool]] = []
    
    def add_validator(self, validator: Callable[[dict], bool]) -> None:
        """Add a validation function to the processing pipeline."""
        self._validators.append(validator)
    
    def process_item(self, item: dict[str, str]) -> dict[str, str] | None:
        """Process a single item through validation and transformation."""
        # Validate item using all registered validators
        if not all(validator(item) for validator in self._validators):
            logger.warning(f"Item failed validation: {item}")
            return None
            
        # Apply transformations based on configuration
        return self._transform_item(item)
```

**Architecture Guidelines:**
- Follow the Single Responsibility Principle - each class and function should have one clear purpose
- Apply DRY (Don't Repeat Yourself) - extract common functionality into reusable components
- Design small, focused functions that are easy to test and understand
- Structure code to prevent circular dependencies and promote loose coupling

## Testing Excellence

Generate comprehensive tests that ensure code reliability:

```python
# tests/test_data_processor.py
import pytest
from unittest.mock import Mock, patch

from my_package.processor import DataProcessor

class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    @pytest.fixture
    def processor(self) -> DataProcessor:
        """Create a DataProcessor instance for testing."""
        config = {"transform_mode": "standard"}
        return DataProcessor(config)
    
    def test_process_item_with_valid_data(self, processor: DataProcessor) -> None:
        """Test that valid items are processed correctly."""
        # Given
        valid_item = {"id": "123", "name": "test"}
        
        # When
        result = processor.process_item(valid_item)
        
        # Then
        assert result is not None
        assert result["id"] == "123"
    
    def test_process_item_with_failing_validator(self, processor: DataProcessor) -> None:
        """Test that items failing validation return None."""
        # Given
        failing_validator = Mock(return_value=False)
        processor.add_validator(failing_validator)
        invalid_item = {"id": "invalid"}
        
        # When
        result = processor.process_item(invalid_item)
        
        # Then
        assert result is None
        failing_validator.assert_called_once_with(invalid_item)
```

**Testing Standards:**
When generating tests, adhere to these principles:
- Use `pytest` framework for all test generation
- Name test files `test_*.py` and place them in the `tests/` directory
- Write descriptive test names that explain the scenario being tested
- Use fixtures for test setup and dependency injection
- Include both positive and negative test cases

## Anti-Patterns to Avoid

Prevent common Python pitfalls that lead to bugs and maintenance issues:

**Mutable Default Arguments:**
```python
# DON'T DO THIS
def process_items(items, results=[]):  # Dangerous mutable default
    results.append(len(items))
    return results

# DO THIS INSTEAD
def process_items(items: list[str], results: list[int] | None = None) -> list[int]:
    if results is None:
        results = []
    results.append(len(items))
    return results
```

**Silent Error Suppression:**
```python
# DON'T DO THIS
try:
    risky_operation()
except:
    pass  # Silent failure - problems will hide

# DO THIS INSTEAD
try:
    risky_operation()
except SpecificError as e:
    logger.warning(f"Expected error in risky_operation: {e}")
    # Handle appropriately or re-raise
```

**Overly Complex Comprehensions:**
```python
# DON'T DO THIS (hard to read and debug)
result = [transform(x) for x in items if complex_condition(x) and other_check(x, y) for y in get_related(x)]

# DO THIS INSTEAD (clear and maintainable)
result = []
for item in items:
    if complex_condition(item):
        related_items = get_related(item)
        for related in related_items:
            if other_check(item, related):
                result.append(transform(item))
```

## Implementation Principles

When generating code, prioritize:

1. **Clarity over cleverness** - Write code that is immediately understandable
2. **Robustness** - Handle edge cases and potential failures gracefully  
3. **Maintainability** - Structure code so it can be easily modified and extended
4. **Performance awareness** - Write efficient code without premature optimization
5. **Security mindset** - Consider potential security implications of the code

Apply these guidelines consistently to create Python code that is professional, reliable, and ready for production use.