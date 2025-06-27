"""PyTorch Lightning module for EA-LSTM model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from .config import EALSTMConfig
from .model import EALSTM, BiEALSTM


class LitEALSTM(BaseLitModel):
    """PyTorch Lightning Module implementation of EA-LSTM.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the EA-LSTM model within our hydrological forecasting framework.
    """

    def __init__(
        self,
        config: EALSTMConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitEALSTM module.

        Args:
            config: EA-LSTM configuration as an EALSTMConfig instance or dict
        """
        # Convert dict config to EALSTMConfig if needed
        ealstm_config = EALSTMConfig.from_dict(config) if isinstance(config, dict) else config

        # Initialize base lightning model with the config
        super().__init__(ealstm_config)

        # Create the appropriate model based on configuration
        if ealstm_config.bidirectional:
            self.model = BiEALSTM(ealstm_config)
        else:
            self.model = EALSTM(ealstm_config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the EA-LSTM model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer with potentially different learning rates for model components.

        For bidirectional models, this allows tuning learning rates for past and future branches separately.

        Returns:
            Dictionary with optimizer and learning rate scheduler configuration
        """
        # Default to standard optimizer configuration from base class
        if not hasattr(self.config, "bidirectional") or not self.config.bidirectional:
            return super().configure_optimizers()

        # For bidirectional model, create parameter groups with custom learning rates
        param_groups = [
            {
                "params": self.model.past_ealstm.parameters(),
                "lr": self.config.learning_rate,
            },
            {
                "params": self.model.projection.parameters(),
                "lr": self.config.learning_rate,
            },
        ]

        # Add future branch parameters, potentially with different learning rate
        future_lr_factor = getattr(self.config, "future_lr_factor", 1.0)
        param_groups.append(
            {
                "params": self.model.future_ealstm.parameters(),
                "lr": self.config.learning_rate * future_lr_factor,
            }
        )

        optimizer = torch.optim.Adam(param_groups)

        # Create scheduler dictionary
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
