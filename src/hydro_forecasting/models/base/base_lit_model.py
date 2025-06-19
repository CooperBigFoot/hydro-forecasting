from typing import Any

import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..layers.rev_in import RevIN
from .base_config import BaseConfig


class BaseLitModel(pl.LightningModule):
    """Base Lightning module for all hydrological forecasting models with RevIN support."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize with a model configuration."""
        super().__init__()

        self.config = config
        self.save_hyperparameters(config.to_dict())
        self.mse_criterion = MSELoss()
        self.test_outputs = []
        self.test_results = None

        # Initialize RevIN layer if enabled
        if getattr(config, "use_rev_in", True):
            self.rev_in = RevIN(num_features=1, eps=1e-5, affine=True)
        else:
            self.rev_in = None

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def _apply_rev_in_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RevIN normalization to the target feature only.

        Args:
            x: Input tensor [batch_size, input_len, input_size]

        Returns:
            Tensor with normalized target feature and unchanged other features
        """
        if self.rev_in is None:
            return x

        # Extract target feature (first column)
        x_target = x[:, :, 0:1]  # [batch_size, input_len, 1]

        # Apply RevIN normalization to target only
        x_target_normalized = self.rev_in(x_target, mode="norm")

        # Reconstruct X with normalized target + unchanged other features
        if x.size(-1) > 1:
            x_other_features = x[:, :, 1:]
            x_normalized = torch.cat([x_target_normalized, x_other_features], dim=-1)
        else:
            x_normalized = x_target_normalized

        return x_normalized

    def _apply_rev_in_denormalization(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Apply RevIN denormalization to predictions.

        Args:
            y_hat: Model predictions [batch_size, output_len, 1]

        Returns:
            Denormalized predictions
        """
        if self.rev_in is None:
            return y_hat

        # Apply RevIN denormalization
        y_hat_denormalized = self.rev_in(y_hat, mode="denorm")

        return y_hat_denormalized

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step with RevIN integration."""
        # Extract inputs
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch.get("static")
        future = batch.get("future")

        # Apply RevIN normalization to input if enabled
        x_processed = self._apply_rev_in_normalization(x)

        # Forward pass with potentially normalized input
        y_hat = self(x_processed, static, future)

        # Apply RevIN denormalization to predictions if enabled
        y_hat_final = self._apply_rev_in_denormalization(y_hat)

        # Calculate loss with denormalized predictions and original targets
        loss = self._compute_loss(y_hat_final, y)

        # Log metrics
        self.log("train_loss", loss, batch_size=x.size(0))

        return loss

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        return self.mse_criterion(predictions, targets)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Execute validation step with RevIN integration."""
        # Extract inputs
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch.get("static")
        future = batch.get("future")

        # Apply RevIN normalization to input if enabled
        x_processed = self._apply_rev_in_normalization(x)

        # Forward pass with potentially normalized input
        y_hat = self(x_processed, static, future)

        # Apply RevIN denormalization to predictions if enabled
        y_hat_final = self._apply_rev_in_denormalization(y_hat)

        # Calculate loss with denormalized predictions and original targets
        loss = self._compute_loss(y_hat_final, y)

        # Log metrics
        self.log("val_loss", loss, batch_size=x.size(0))

        return {"val_loss": loss, "preds": y_hat_final, "targets": y}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Execute test step with RevIN integration."""
        # Extract inputs
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch.get("static")
        future = batch.get("future")

        # Apply RevIN normalization to input if enabled
        x_processed = self._apply_rev_in_normalization(x)

        # Forward pass with potentially normalized input
        y_hat = self(x_processed, static, future)

        # Apply RevIN denormalization to predictions if enabled
        y_hat_final = self._apply_rev_in_denormalization(y_hat)

        # Calculate loss with denormalized predictions and original targets
        loss = self._compute_loss(y_hat_final, y)

        # Log metrics
        self.log("test_loss", loss, batch_size=x.size(0))

        # Create standardized output dictionary
        output = {
            "predictions": y_hat_final.squeeze(-1),
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.config.group_identifier],
        }

        # Add optional fields if present in batch
        for field in ["input_end_date", "slice_idx"]:
            if field in batch:
                output[field] = batch[field]

        # Store output for later processing
        self.test_outputs.append(output)

        return output

    def on_test_epoch_start(self) -> None:
        """Reset test outputs at start of test epoch."""
        self.test_outputs = []
        # Reset RevIN statistics for test epoch
        if self.rev_in is not None:
            self.rev_in.reset_statistics()

    def on_test_epoch_end(self) -> None:
        """Process test outputs at end of test epoch."""
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Consolidate outputs
        self.test_results = {
            "predictions": torch.cat([o["predictions"] for o in self.test_outputs]),
            "observations": torch.cat([o["observations"] for o in self.test_outputs]),
            "basin_ids": [bid for o in self.test_outputs for bid in o["basin_ids"]],
        }

        # Add optional fields if present
        for field in ["input_end_date", "slice_idx"]:
            if field in self.test_outputs[0]:
                self.test_results[field] = [item for o in self.test_outputs for item in o[field]]

        # Clean up temporary storage
        self.test_outputs = []

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)

        # Create scheduler dictionary
        scheduler_config = {
            "scheduler": ReduceLROnPlateau(
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
