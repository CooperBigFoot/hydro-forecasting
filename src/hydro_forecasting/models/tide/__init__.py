"""TiDE model implementation module.

TiDE (Time-series Dense Encoder) is a model architecture based on the paper:
"Long-term Forecasting with TiDE: Time-series Dense Encoder"
https://arxiv.org/pdf/2304.08424

This module provides implementations for:
1. TiDEConfig - Configuration class for TiDE models
2. TiDEModel - PyTorch implementation of the TiDE architecture
3. LitTiDE - PyTorch Lightning wrapper for training and evaluation
"""

from .config import TiDEConfig
from .lightning import LitTiDE
from .model import TiDEModel, TiDEResBlock

__all__ = [
    "TiDEConfig",
    "TiDEModel",
    "TiDEResBlock",
    "LitTiDE",
]
