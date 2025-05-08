"""
Configuration for the RepeatLastValues model, a simple baseline that repeats
the last observed value for the entire forecast horizon.
"""

from typing import ClassVar

from ..base.base_config import BaseConfig


class RepeatLastValuesConfig(BaseConfig):
    """Configuration class for RepeatLastValues model.

    RepeatLastValues is a simple baseline model that forecasts by repeating
    the last observed value of the target variable for the entire forecast horizon.
    """

    # This model has no specific parameters beyond the standard ones
    MODEL_PARAMS: ClassVar[list[str]] = []

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        hidden_size: int = 8,  # Not used but kept for API compatibility
        dropout: float = 0.0,  # Not used but kept for API compatibility
        learning_rate: float = 1e-10,  # Very small since this model doesn't learn
        group_identifier: str = "gauge_id",
        **kwargs,
    ):
        """Initialize RepeatLastValues configuration.

        Args:
            input_len: Length of the input sequence
            output_len: Length of the output sequence (forecast horizon)
            input_size: Number of input features (target + past features)
            static_size: Number of static features (not used)
            future_input_size: Number of future forcing features (not used)
            hidden_size: Size of hidden layers (not used in this model)
            dropout: Dropout rate (not used in this model)
            learning_rate: Initial learning rate (minimal since model doesn't train)
            group_identifier: Name of the column identifying catchment groups
            **kwargs: Additional parameters
        """
        # Initialize base config with standard parameters
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            future_input_size=future_input_size,
            learning_rate=learning_rate,
            group_identifier=group_identifier,
            **kwargs,
        )

        # Set only model-specific parameters
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Validate parameters
        if input_size < 1:
            raise ValueError("input_size must be at least 1")
        if output_len < 1:
            raise ValueError("output_len must be at least 1")
