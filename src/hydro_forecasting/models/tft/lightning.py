from typing import Any

import torch
import torch.nn as nn

from ..base.base_lit_model import BaseLitModel
from .config import TFTConfig
from .model import TemporalFusionTransformer


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting.

    This loss function penalizes underestimation and overestimation
    differently based on the specified quantile.
    """

    def __init__(self, quantiles: list[float]):
        """
        Initialize quantile loss.

        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            preds: Predictions with shape [batch_size, seq_len, n_quantiles]
            target: Target values with shape [batch_size, seq_len, 1]

        Returns:
            Quantile loss value
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i : i + 1]
            losses.append(torch.max((q - 1) * errors, q * errors))

        # Sum across quantiles and average across batch and time
        loss = torch.cat(losses, dim=-1).mean()
        return loss


class LitTFT(BaseLitModel):
    """PyTorch Lightning Module implementation of TFT.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the Temporal Fusion Transformer within our
    hydrological forecasting framework.
    """

    def __init__(
        self,
        config: TFTConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitTFT module.

        Args:
            config: TFT configuration as a TFTConfig instance or dict
        """
        # Convert dict config to TFTConfig if needed
        if isinstance(config, dict):
            config = TFTConfig.from_dict(config)

        # Initialize base lightning model with the config
        super().__init__(config)

        # Create the TFT model
        self.model = TemporalFusionTransformer(config)

        # Set up quantile loss if needed
        if hasattr(config, "quantiles") and len(config.quantiles) > 1:
            self.quantile_loss = QuantileLoss(config.quantiles)
            self.use_quantile_loss = True
        else:
            self.use_quantile_loss = False

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TFT model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, n_quantiles]
        """
        return self.model(x, static, future)

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets.

        Args:
            predictions: Model predictions [batch_size, output_len, n_quantiles]
            targets: Ground truth values [batch_size, output_len, 1]

        Returns:
            Loss value
        """
        if self.use_quantile_loss:
            return self.quantile_loss(predictions, targets)
        else:
            # If not using quantile loss, use MSE (default from BaseLitModel)
            return super()._compute_loss(predictions, targets)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step with additional logging.

        Args:
            batch: Input batch dictionary
            batch_idx: Batch index

        Returns:
            Loss value
        """
        # Use the base training step implementation
        loss = super().training_step(batch, batch_idx)

        # Additional logging for TFT-specific metrics
        if (
            batch_idx % 100 == 0
            and hasattr(self.model, "attention")
            and hasattr(self.model.attention, "attn_output_weights")
        ):
            attn_weights = self.model.attention.attn_output_weights
            if attn_weights is not None:
                self.log("train_attention_score", attn_weights.mean(), on_step=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer with appropriate parameters.

        Returns:
            Dictionary with optimizer and learning rate scheduler configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

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
