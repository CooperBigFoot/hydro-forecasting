"""
Adapted RevIN (Reversible Instance Normalization) implementation for hydrological time series forecasting.
Original implementation: https://github.com/ts-kim/RevIN/blob/master/RevIN.py#L4

Based on the paper: "Reversible Instance Normalization for Accurate Time-Series Forecasting
against Distribution Shift" (Kim et al., 2022)

This implementation is specifically adapted to work with the target feature only and handle
the batch structure used in the hydrological forecasting framework.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series forecasting.

    This implementation only normalizes the target feature (first feature) in the input,
    leaving other features unchanged. It's designed to work with the hydrological
    forecasting framework's data structure.
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True):
        """Initialize RevIN layer.

        Args:
            num_features: Number of features to normalize (typically 1 for target only)
            eps: Small value added for numerical stability
            affine: Whether to use learnable affine transformation
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # Initialize learnable parameters if affine transformation is enabled
        if self.affine:
            self._init_params()

        # Storage for normalization statistics
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply RevIN normalization or denormalization.

        Args:
            x: Input tensor [batch_size, seq_len, features] or [batch_size, seq_len, 1]
            mode: Either 'norm' for normalization or 'denorm' for denormalization

        Returns:
            Processed tensor with same shape as input
        """
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented. Use 'norm' or 'denorm'.")

    def _init_params(self):
        """Initialize learnable affine parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """Compute and store instance-wise statistics.

        Args:
            x: Input tensor [batch_size, seq_len, features]
        """
        # Compute statistics only over the temporal dimension (dimension 1)
        # Keep batch and feature dimensions intact
        dim2reduce = (1,)  # Only reduce over sequence length dimension

        # Compute mean and standard deviation
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input tensor.

        Args:
            x: Input tensor [batch_size, seq_len, features]

        Returns:
            Normalized tensor
        """
        # Normalize: (x - mean) / std
        x_norm = (x - self.mean) / self.stdev

        # Apply learnable affine transformation if enabled
        if self.affine:
            # Reshape affine parameters to match tensor dimensions
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            x_norm = x_norm * weight + bias

        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply denormalization to restore original scale.

        Args:
            x: Normalized tensor [batch_size, seq_len, features]

        Returns:
            Denormalized tensor
        """
        if self.mean is None or self.stdev is None:
            raise RuntimeError("Cannot denormalize: statistics not computed. Call forward with mode='norm' first.")

        # Reverse affine transformation if enabled
        if self.affine:
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            x = (x - bias) / (weight + self.eps * self.eps)

        # Denormalize: x * std + mean
        x_denorm = x * self.stdev + self.mean

        return x_denorm

    def reset_statistics(self):
        """Reset stored statistics. Useful when switching between datasets."""
        self.mean = None
        self.stdev = None
