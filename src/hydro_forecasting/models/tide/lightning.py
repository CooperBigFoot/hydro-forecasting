"""PyTorch Lightning module for TiDE model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from .config import TiDEConfig
from .model import TiDEModel


class LitTiDE(BaseLitModel):
    """PyTorch Lightning Module implementation of TiDE.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the TiDE model within our hydrological forecasting framework.
    """

    def __init__(
        self,
        config: TiDEConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitTiDE module.

        Args:
            config: TiDE configuration as a TiDEConfig instance or dict
        """
        # Convert dict config to TiDEConfig if needed
        tide_config = TiDEConfig.from_dict(config) if isinstance(config, dict) else config

        # Initialize base lightning model with the config
        super().__init__(tide_config)

        # Create the TiDE model
        self.model = TiDEModel(tide_config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TiDE model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)
