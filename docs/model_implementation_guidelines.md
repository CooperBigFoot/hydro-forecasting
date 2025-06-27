# Model Implementation Conventions

This document defines the conventions for implementing and extending models in our hydrological time series analysis project. These guidelines ensure consistency, maintainability, and compatibility when refactoring code or introducing new models. All models in this project should adhere to these standards, including architectures like TSMixer, TiDE, LSTM, RepeatLastValues, and others.

## 1. Overall Structure

Every model implementation should consist of three key components:

1. **Configuration Class**: Extends `BaseConfig` to encapsulate all hyperparameters and model-specific settings as the single source of truth for configuration values.

2. **Core Model (`nn.Module`)**: Implements the main computational logic using PyTorch's `nn.Module`, potentially including multiple submodules for different architectural components.

3. **PyTorch Lightning Module**: Extends `BaseLitModel` to handle training, validation, testing, and logging, serving as the interface between the model and the training pipeline.

> **Note**: All configuration details must be managed exclusively through the configuration class, never hardcoded in the model implementation.

## 2. Model Configuration

### Purpose

The configuration class centralizes all hyperparameters required by the model, ensuring that model-specific details (dimensions, layer counts, learning rates, etc.) are defined in one place.

### Implementation Guidelines

- Extend the `BaseConfig` class for each model (e.g., `TSMixerConfig`, `TiDEConfig`).
- Define a `MODEL_PARAMS` class variable listing model-specific parameters.
- Override `__init__` to include model-specific parameters while passing standard parameters to the superclass.
- Use appropriate type hints for all attributes.
- Implement validation logic for interdependent parameters.

### Standard Hyperparameters

All standard parameters are already defined in `BaseConfig`:

- `input_len`: Length of the historical input sequence (lookback window)
- `output_len`: Length of the forecast horizon (prediction steps)
- `input_size`: Dimensionality of input features per time step (dynamic features)
- `static_size`: Dimensionality of static/time-invariant features
- `future_input_size`: Dimensionality of future forcing features (when applicable)
- `hidden_size`: Size of hidden layers in the model
- `learning_rate`: Initial learning rate for optimization
- `dropout`: Dropout rate for regularization
- `group_identifier`: Column name identifying the grouping variable (e.g., "gauge_id")

### Example Configuration Class

```python
class TSMixerConfig(BaseConfig):
    """Configuration for TSMixer model."""

    # Define model-specific parameters
    MODEL_PARAMS = [
        "static_embedding_size",
        "num_mixing_layers",
        "scheduler_patience",
        "scheduler_factor",
        "fusion_method",
    ]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: Optional[int] = None,
        hidden_size: int = 64,
        static_embedding_size: int = 10,
        num_mixing_layers: int = 5,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 2,
        scheduler_factor: float = 0.5,
        fusion_method: str = "add",
        **kwargs,
    ):
        """Initialize TSMixer configuration."""
        # Initialize base config with standard parameters
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            future_input_size=future_input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            group_identifier=group_identifier,
            **kwargs,
        )

        # Set model-specific parameters
        self.static_embedding_size = static_embedding_size
        self.num_mixing_layers = num_mixing_layers
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.fusion_method = fusion_method

        # Validate parameters
        if fusion_method not in ["add", "concat"]:
            raise ValueError(f"Invalid fusion_method: {fusion_method}. Must be 'add' or 'concat'.")
```

## 3. Core Model (`nn.Module`) Implementation

### Purpose

The core model implements the model's architecture and computational logic, encapsulating all layers, operations, and forward pass behavior.

### Implementation Guidelines

- Break down complex models into clear, modular sub-components using separate classes (e.g., encoder, decoder, mixing blocks).
- Use clear, descriptive names for layers and operations.
- Ensure the forward method accepts inputs for:
  - Historical data (dynamic features)
  - Static features (when applicable)
  - Future forcing features (when applicable)
- Document tensor shapes and dimensional assumptions in docstrings.

### Standard Forward Method Signature

All core models should implement a forward method with this signature:

```python
def forward(
    self, 
    x: torch.Tensor,               # [batch_size, input_len, input_size]
    static: Optional[torch.Tensor] = None,  # [batch_size, static_size]
    future: Optional[torch.Tensor] = None,  # [batch_size, output_len, future_input_size]
) -> torch.Tensor:                 # [batch_size, output_len, 1]
    """Forward pass implementation."""
    pass
```

## 3a. Data Interface Expectations

Models in our framework interact with standardized data provided by the `HydroDataModule` and `HydroDataset` classes. Understanding this interface is essential for implementing compatible models.

### Batch Structure

All models should expect batches with the following structure:

```python
{
    "X": torch.Tensor,              # [batch_size, input_len, input_size] - Historical time series data
    "y": torch.Tensor,              # [batch_size, output_len] - Target values to predict
    "static": torch.Tensor,         # [batch_size, static_size] - Static catchment attributes
    "future": torch.Tensor,         # [batch_size, output_len, future_input_size] - Future forcing data (optional)
    "gauge_id": List[str],          # Basin identifiers
    "slice_idx": List[List[int]],   # Original indices in the dataset
    "input_end_date": List[str],    # End dates of input windows
    "domain_id": torch.Tensor,      # [batch_size, 1] - Domain identifier (for transfer learning)
    "domain_name": str              # Domain name (for transfer learning)
}
```

Note that not all fields will be present in every batch. Models should handle cases where optional elements (particularly `future`) are missing.

### Data Characteristics

- **Historical Data (`X`)**: Contains the target variable and potentially other dynamic features for the input window. The first feature is always the target variable.
- **Target Values (`y`)**: Contains the target values for the forecast horizon. During training, these are the ground truth values to predict.
- **Static Features (`static`)**: Contains time-invariant catchment attributes preprocessed as tensors.
- **Future Forcing (`future`)**: When available, contains known or forecasted external variables for the prediction period. Not all datasets will provide this.

### Handling Missing Components

Models should implement graceful fallbacks when optional components are missing:

```python
def forward(
    self, 
    x: torch.Tensor, 
    static: torch.Tensor, 
    future: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Forward pass with robust handling of optional components."""
    # Handle case where future forcing is not available
    if future is None:
        # Implement fallback behavior
        pass
    
    # Rest of implementation
    ...
```

### Preprocessing Considerations

- All features are preprocessed by the `HydroDataModule` according to the configured pipeline.
- Models receive standardized data (typically z-score normalized) and should output predictions in the same scale.
- The `HydroDataModule` handles inverse transformations when evaluating performance metrics.

### Transfer Learning Support

For models supporting transfer learning:

- The `domain_id` tensor indicates whether each sample is from the source (0.0) or target (1.0) domain.
- The `domain_name` provides a string identifier for the specific domain (e.g., "CH" for Switzerland).
- Models can use this information to implement domain-specific processing or domain adaptation techniques.

By designing models to work with this standardized interface, we ensure compatibility with the training pipeline and facilitate easier comparison between different architectures.

## 4. PyTorch Lightning Module Wrapper

### Purpose

The Lightning module wraps the core model and handles training logic, loss computation, metric logging, and optimization.

### Implementation Guidelines

#### Class Structure

- Extend `BaseLitModel` for all model implementations.
- Accept either a configuration object or a dictionary in the constructor.
- Create an instance of the corresponding core model in the constructor.
- Override model-specific methods only when needed (most functionality is provided by the base class).

#### Default Methods Provided by BaseLitModel

The `BaseLitModel` already implements these methods, which typically don't need to be overridden:

- `training_step`: Handles data extraction, forward pass, loss computation, and logging
- `validation_step`: Similar to training_step with validation metrics
- `test_step`: Similar to validation_step with standardized output collection
- `configure_optimizers`: Sets up optimizer and learning rate scheduler
- `_compute_loss`: Calculates loss using MSE by default

#### Custom Methods for Model-Specific Logic

Override these methods only when model-specific logic is needed:

- `forward`: Must be implemented to delegate to the core model
- `_log_additional_train_metrics`: Optional to add model-specific metrics during training
- `_log_additional_val_metrics`: Optional to add model-specific metrics during validation

#### Example Lightning Module

```python
class LitTSMixer(BaseLitModel):
    """PyTorch Lightning Module implementation of TSMixer."""

    def __init__(
        self,
        config: Union[TSMixerConfig, Dict[str, Any]],
    ):
        """Initialize the Lightning Module with a TSMixerConfig."""
        # Convert dict config to TSMixerConfig if needed
        if isinstance(config, dict):
            config = TSMixerConfig.from_dict(config)

        # Initialize base class with the config
        super().__init__(config)

        # Create the TSMixer model using the config
        self.model = TSMixer(config)

    def forward(
        self, 
        x: torch.Tensor, 
        static: Optional[torch.Tensor] = None, 
        future: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass that delegates to the TSMixer model."""
        return self.model(x, static, future)
        
    # Optional: Add model-specific customizations
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.log("info", "Backbone parameters frozen")
```

## 5. Hyperparameter Naming Conventions

To maintain clarity and consistency, use these naming conventions for hyperparameters:

### Lengths and Dimensions

- `input_len`: Length of the historical input sequence (time steps).
- `output_len`: Forecast horizon or number of future steps to predict.
- `input_size`: Dimensionality of input features per time step.
- `static_size`: Dimensionality of static (time-invariant) features.
- `future_input_size`: Dimensionality of future forcing features (when applicable).
- `hidden_size`: Size of hidden layers in the model.

### Other Parameters

- Use the suffix `_size` for dimensions of vector spaces.
- Use the suffix `_len` for sequence lengths or time steps.
- Use the suffix `_dim` only when referring to specific dimensions in a tensor.
- Use descriptive prefixes to differentiate sizes for different components (e.g., `encoder_hidden_size`, `decoder_hidden_size`).

## 6. Feature Terminology

For clarity and consistency, use these definitions throughout the codebase:

- **Target**: The primary time series variable being forecast (typically streamflow).
- **Dynamic Features**: Time-varying features that include the target and potentially other time series.
- **Static Features**: Time-invariant features that describe fixed properties (e.g., catchment attributes).
- **Future Forcing Features**: Known or forecasted external variables for the prediction period.
- **Forcing Features**: External inputs (not the target) used to improve the forecast, either historical or future.

## 7. Logging Conventions

All models must consistently log these metrics:

### Required Metrics

- `train_loss`: Loss value during training.
- `val_loss`: Loss value during validation.

### Additional Metrics

Models may log additional metrics like:

- `train_rmse`: Root mean squared error during training.
- `val_rmse`: Root mean squared error during validation.
- `learning_rate`: Current learning rate (if using a scheduler).

### Metric Naming Convention

- Use the format `{phase}_{metric}` where:
  - `phase` is one of: `train`, `val`, `test`
  - `metric` describes the measurement: `loss`, `mse`, `rmse`, etc.

## 8. Model-Agnostic Training Script

A model-agnostic training script should:

1. Take a model configuration file as input.
2. Load and prepare data using standard data modules.
3. Instantiate the appropriate model class based on the configuration.
4. Set up training parameters, callbacks, and loggers.
5. Train the model and evaluate performance.
6. Save model checkpoints and evaluation results.

Specific implementation details will be added in a future update.

## 9. Code Style and Documentation

### Code Style

- Follow Ruff standards for Python code formatting.
- Use consistent indentation (4 spaces).
- Use meaningful variable names that reflect hydrological domain concepts.

### Documentation

- Include comprehensive docstrings for all classes and methods.
- Follow Google docstring style:

  ```python
  def function(arg1, arg2):
      """Short description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1.
          arg2: Description of arg2.
          
      Returns:
          Description of return value.
          
      Raises:
          ExceptionType: When and why this exception is raised.
      """
  ```

- Document expected tensor shapes in the forward method.
- Include examples for complex methods or classes.

### Type Hints

- Use type hints for all function and method arguments.
- Use generic types (e.g., `List`, `Dict`, `Optional`) from the `typing` module.
- Use `Union` for parameters that can accept multiple types.

## 10. Testing and Error Handling

### Error Handling

- Validate input shapes and types in forward methods.
- Provide clear error messages that help identify the source of problems.
- Add assertions for critical assumptions and invariants.

## 11. Extensions and Special Cases

This section will be expanded as new requirements or special cases are identified. It may include guidelines for:

- Transfer learning capabilities
- Handling missing data
- Model interpretability features
- Uncertainty quantification
