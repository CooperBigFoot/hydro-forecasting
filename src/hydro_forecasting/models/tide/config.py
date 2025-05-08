from typing import ClassVar

from ..base.base_config import BaseConfig


class TiDEConfig(BaseConfig):
    """Configuration class for TiDE model.

    TiDE (Time-series Dense Encoder) is a model architecture for time series forecasting
    that uses dense residual networks to process historical data, future forcing features,
    and static features.

    Reference: "Long-term Forecasting with TiDE: Time-series Dense Encoder"
    https://arxiv.org/pdf/2304.08424
    """

    # Define model-specific parameters - removed hidden_size and dropout as they should be standard
    MODEL_PARAMS: ClassVar[list[str]] = [
        "num_encoder_layers",
        "num_decoder_layers",
        "decoder_output_size",
        "temporal_decoder_hidden_size",
        "past_feature_projection_size",
        "future_forcing_projection_size",
        "use_layer_norm",
        "scheduler_patience",
        "scheduler_factor",
        "hidden_size",
        "dropout",
    ]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        hidden_size: int = 128,
        dropout: float = 0.1,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_size: int = 16,
        temporal_decoder_hidden_size: int = 32,
        past_feature_projection_size: int = 0,
        future_forcing_projection_size: int = 0,
        use_layer_norm: bool = True,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        **kwargs,
    ):
        """Initialize TiDE configuration.

        Args:
            input_len: Length of the input sequence
            output_len: Length of the output sequence (forecast horizon)
            input_size: Number of input features (target + past features)
            static_size: Number of static features
            future_input_size: Number of future forcing features
            hidden_size: Size of hidden layers
            dropout: Dropout rate
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            decoder_output_size: Size of decoder output embedding
            temporal_decoder_hidden_size: Hidden size in temporal decoder
            past_feature_projection_size: Projection size for past features (0 = no projection)
            future_forcing_projection_size: Projection size for future features (0 = no projection)
            use_layer_norm: Whether to use layer normalization
            learning_rate: Initial learning rate
            group_identifier: Name of the column identifying catchment groups
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            **kwargs: Additional parameters
        """
        # Initialize base config with standard parameters
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

        # Set only model-specific parameters that aren't in standard params
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.static_size = static_size
        self.future_input_size = future_input_size or input_size - 1
        self.past_feature_projection_size = past_feature_projection_size
        self.future_forcing_projection_size = future_forcing_projection_size
        self.use_layer_norm = use_layer_norm
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_size = decoder_output_size
        self.temporal_decoder_hidden_size = temporal_decoder_hidden_size
        self.past_feature_projection_size = past_feature_projection_size
        self.future_forcing_projection_size = future_forcing_projection_size
        self.use_layer_norm = use_layer_norm
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # Validate parameters
        if self.num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be at least 1")
        if self.num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be at least 1")
