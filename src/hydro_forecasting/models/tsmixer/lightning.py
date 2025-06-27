"""PyTorch Lightning module for TSMixer model."""

from typing import Any

import torch
from torch.optim import Adam

from ..base.base_lit_model import BaseLitModel
from .config import TSMixerConfig
from .model import TSMixer


class LitTSMixer(BaseLitModel):
    """PyTorch Lightning Module implementation of TSMixer.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the TSMixer model within our hydrological forecasting
    framework.
    """

    def __init__(
        self,
        config: TSMixerConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitTSMixer module.

        Args:
            config: TSMixer configuration as a TSMixerConfig instance or dict
        """
        # Convert dict config to TSMixerConfig if needed
        tsmixer_config = TSMixerConfig.from_dict(config) if isinstance(config, dict) else config

        # Initialize base lightning model with the config
        super().__init__(tsmixer_config)

        # Create the underlying TSMixer model
        self.model = TSMixer(tsmixer_config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TSMixer model.

        Args:
            x: Historical input features [B, input_len, input_size]
            static: Static features [B, static_size]
            future: Optional future forcing data [B, output_len, future_input_size]

        Returns:
            Predictions [B, output_len, 1]
        """
        return self.model(x, static, future)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.model.mixing_stage.parameters():
            param.requires_grad = False
        self.log("info", "Backbone parameters frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.model.mixing_stage.parameters():
            param.requires_grad = True
        self.log("info", "Backbone parameters unfrozen")

    def freeze_head(self) -> None:
        """Freeze prediction head parameters."""
        for param in self.model.head.parameters():
            param.requires_grad = False
        self.log("info", "Head parameters frozen")

    def unfreeze_head(self) -> None:
        """Unfreeze prediction head parameters."""
        for param in self.model.head.parameters():
            param.requires_grad = True
        self.log("info", "Head parameters unfrozen")

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and lr_scheduler configuration
        """
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)

        # Create scheduler dictionary
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=getattr(self.config, "scheduler_patience", 5),
                factor=getattr(self.config, "scheduler_factor", 0.5),
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
