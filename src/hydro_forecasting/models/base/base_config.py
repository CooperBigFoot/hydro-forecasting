from typing import Any, ClassVar


class BaseConfig:
    """Base configuration class for all hydrological forecasting models."""

    # Standard parameters all models should support
    STANDARD_PARAMS: ClassVar[list[str]] = [
        "input_len",
        "output_len",
        "input_size",
        "static_size",
        "future_input_size",
        "learning_rate",
        "group_identifier",
        "use_rev_in",
    ]

    # Model-specific parameters to be defined in subclasses
    MODEL_PARAMS: ClassVar[list[str]] = []

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        learning_rate: float = 1e-5,
        group_identifier: str = "gauge_id",
        use_rev_in: bool = True,
        **kwargs,
    ):
        """Initialize base configuration with standard parameters."""
        # Set standard parameters
        self.input_len = input_len
        self.output_len = output_len
        self.input_size = input_size
        self.static_size = static_size
        self.future_input_size = future_input_size or max(1, input_size - 1)
        self.learning_rate = learning_rate
        self.group_identifier = group_identifier
        self.use_rev_in = use_rev_in  # Added RevIN configuration

        # Set model-specific parameters
        self._set_model_params(**kwargs)

    def _set_model_params(self, **kwargs):
        """Set model-specific parameters with validation."""
        for param, value in kwargs.items():
            if param in self.MODEL_PARAMS:
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown parameter '{param}' for {self.__class__.__name__}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BaseConfig":
        """Create a config object from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def update(self, **kwargs) -> "BaseConfig":
        """Update config parameters."""
        for key, value in kwargs.items():
            if key in self.STANDARD_PARAMS or key in self.MODEL_PARAMS:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter '{key}' for {self.__class__.__name__}")
        return self
