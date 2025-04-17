import pytorch_lightning as pl
import torch
from typing import Dict, Optional, Any
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .base_config import BaseConfig


class BaseLitModel(pl.LightningModule):
    """Base Lightning module for all hydrological forecasting models."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize with a model configuration."""
        super().__init__()

        self.config = config
        self.save_hyperparameters(config.to_dict())
        self.mse_criterion = MSELoss()
        self.test_outputs = []
        self.test_results = None

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute training step."""
        # Extract inputs
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch.get("static")
        future = batch.get("future")

        # Forward pass
        y_hat = self(x, static, future)

        # Calculate loss
        loss = self._compute_loss(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, batch_size=x.size(0))

        return loss

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        return self.mse_criterion(predictions, targets)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute validation step."""
        # Extract inputs
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch.get("static")
        future = batch.get("future")

        # Forward pass
        y_hat = self(x, static, future)

        # Calculate loss
        loss = self._compute_loss(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, batch_size=x.size(0))

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute test step."""
        # Extract inputs
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch.get("static")
        future = batch.get("future")

        # Forward pass
        y_hat = self(x, static, future)

        # Calculate loss
        loss = self._compute_loss(y_hat, y)

        # Log metrics
        self.log("test_loss", loss, batch_size=x.size(0))

        # Create standardized output dictionary
        output = {
            "predictions": y_hat.squeeze(-1),
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
                self.test_results[field] = [
                    item for o in self.test_outputs for item in o[field]
                ]

        # Clean up temporary storage
        self.test_outputs = []

    def configure_optimizers(self) -> Dict[str, Any]:
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
