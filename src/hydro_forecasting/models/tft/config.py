from typing import ClassVar

from ..base.base_config import BaseConfig


class TFTConfig(BaseConfig):
    """Configuration class for Temporal Fusion Transformer model.

    The Temporal Fusion Transformer (TFT) is an attention-based architecture that combines
    high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.

    Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    https://arxiv.org/abs/1912.09363
    """

    # Define model-specific parameters
    MODEL_PARAMS: ClassVar[list[str]] = [
        "hidden_size",
        "lstm_layers",
        "num_attention_heads",
        "dropout",
        "hidden_continuous_size",
        "attn_dropout",
        "add_relative_index",
        "scheduler_patience",
        "scheduler_factor",
        "quantiles",
        "context_length_ratio",
        "encoder_layers",
    ]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int | None = None,
        attn_dropout: float = 0.0,
        add_relative_index: bool = True,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        quantiles: list[float] = None,
        context_length_ratio: float = 1.0,
        encoder_layers: int = 1,
        **kwargs,
    ):
        """Initialize TFT configuration.

        Args:
            input_len: Length of the input sequence (lookback window)
            output_len: Length of the forecast horizon
            input_size: Number of input features per timestep
            static_size: Number of static features
            future_input_size: Number of future forcing features (defaults to input_size minus 1)
            hidden_size: Size of hidden layers
            lstm_layers: Number of LSTM layers in encoder/decoder
            num_attention_heads: Number of heads in multi-head attention
            dropout: Dropout rate
            hidden_continuous_size: Size of hidden layers for processing continuous variables

            attn_dropout: Dropout rate for attention
            add_relative_index: Whether to add relative index as a feature
            learning_rate: Initial learning rate
            group_identifier: Column name identifying the grouping variable (e.g., "gauge_id")
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            quantiles: Quantiles to predict for probabilistic forecasting
            context_length_ratio: Ratio of context length to input length
            encoder_layers: Number of encoder layers
            **kwargs: Additional parameters
        """
        # Initialize base config with standard parameters
        if quantiles is None:
            quantiles = [0.5]
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            future_input_size=future_input_size,
            learning_rate=learning_rate,
            group_identifier=group_identifier,
            **kwargs,
        )

        # Set model-specific parameters
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size or hidden_size
        self.attn_dropout = attn_dropout
        self.add_relative_index = add_relative_index

        self.quantiles = quantiles
        self.context_length_ratio = context_length_ratio
        self.encoder_layers = encoder_layers

        # Scheduler parameters
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # Validate parameters
        if self.lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1")
