"""EA-LSTM model implementation module.

EA-LSTM (Entity-Aware LSTM) is a model architecture based on the paper:
"Towards learning universal, regional, and local hydrological behaviors via
machine learning applied to large-sample datasets" by Kratzert et al. (2019)
https://hess.copernicus.org/articles/23/5089/2019/

This module provides implementations for:
1. EALSTMConfig - Configuration class for EA-LSTM models
2. EALSTM - PyTorch implementation of the EA-LSTM architecture
3. BiEALSTM - Bidirectional EA-LSTM that processes past and future data
4. LitEALSTM - PyTorch Lightning wrapper for training and evaluation
"""

from .config import EALSTMConfig
from .lightning import LitEALSTM
from .model import EALSTM, BiEALSTM, EALSTMCell

__all__ = [
    "EALSTMConfig",
    "EALSTM",
    "EALSTMCell",
    "BiEALSTM",
    "LitEALSTM",
]
