"""
RepeatLastValues model implementation.

This is a simple baseline model that repeats the last observed value
for the entire forecast horizon.
"""

import torch
import torch.nn as nn

from .config import RepeatLastValuesConfig


class RepeatLastValues(nn.Module):
    """RepeatLastValues model implementation.

    This model predicts future values by repeating the last observed value
    for the entire forecast horizon. It serves as a simple baseline against
    which more complex models can be compared.
    """

    def __init__(self, config: RepeatLastValuesConfig):
        """Initialize RepeatLastValues model.

        Args:
            config: Configuration object for RepeatLastValues model
        """
        super().__init__()
        self.config = config

        # Store config parameters for convenience
        self.input_size = config.input_size
        self.output_len = config.output_len

        # This model has no trainable parameters
        # but we create a dummy parameter to satisfy PyTorch's requirements
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: repeat the last observed value for the entire forecast horizon.

        Args:
            x: Input time series, shape [batch_size, input_len, input_size]
               (contains target as the first feature, followed by optional past features)
            static: Static features [batch_size, static_size] (not used)
            future: Future forcing data [batch_size, output_len, future_input_size] (not used)

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        # Extract the target variable (assumed to be the first feature)
        x_target = x[:, :, 0:1]

        # Extract the last observed value: shape [batch_size, 1, 1]
        last_value = x_target[:, -1:, :]

        # Repeat the last value for the forecast horizon
        predictions = last_value.repeat(1, self.output_len, 1)

        return predictions
