from typing import ClassVar

from ..base.base_config import BaseConfig


class EALSTMConfig(BaseConfig):
    """Configuration class for Entity-Aware LSTM model.

    EA-LSTM is a model architecture that uses static catchment attributes to modulate
    the LSTM input gate, enabling better transfer learning between different catchments.

    Reference: "Kratzert et al. (2019) - Towards learning universal, regional, and
    local hydrological behaviors via machine learning applied to large-sample datasets"
    https://hess.copernicus.org/articles/23/5089/2019/
    """

    # Define model-specific parameters
    MODEL_PARAMS: ClassVar[list[str]] = [
        "num_layers",
        "bias",
        "scheduler_patience",
        "scheduler_factor",
        "dropout",
        "hidden_size",
        # Bidirectional EA-LSTM parameters
        "future_input_size",
        "future_hidden_size",
        "future_layers",
        "bidirectional_fusion",
        "bidirectional",
    ]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        hidden_size: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        num_layers: int = 1,
        bias: bool = True,
        # Bidirectional EA-LSTM parameters
        future_hidden_size: int | None = None,
        future_layers: int | None = None,
        bidirectional_fusion: str = "concat",
        bidirectional: bool = True,
        **kwargs,
    ):
        """Initialize EA-LSTM configuration.

        Args:
            input_len: Length of the input sequence (lookback window)
            output_len: Length of the forecast horizon
            input_size: Number of input features per timestep
            static_size: Number of static/time-invariant features
            future_input_size: Number of future forcing features (defaults to input_size minus 1)
            hidden_size: Size of the LSTM hidden state
            dropout: Dropout rate for regularization
            learning_rate: Initial learning rate for optimization
            group_identifier: Column name identifying the grouping variable (e.g., "gauge_id")
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            num_layers: Number of stacked LSTM layers
            bias: Whether to use bias in LSTM layers
            future_hidden_size: Size of hidden state for future branch (defaults to hidden_size)
            future_layers: Number of layers in future branch (defaults to num_layers)
            bidirectional_fusion: Method for combining past and future states ("concat", "add", "average")
            bidirectional: Whether to use bidirectional model
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

        # Set model-specific parameters
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.future_input_size = future_input_size or input_size - 1
        self.group_identifier = group_identifier
        self.static_size = static_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.bias = bias
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # Bidirectional EA-LSTM parameters
        self.future_hidden_size = future_hidden_size or hidden_size
        self.future_layers = future_layers or num_layers
        self.bidirectional_fusion = bidirectional_fusion
        self.bidirectional = bidirectional

        # Validate parameters
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.future_layers < 1:
            raise ValueError("future_layers must be at least 1")
        if self.bidirectional_fusion not in ["concat", "add", "average"]:
            raise ValueError("bidirectional_fusion must be one of: concat, add, average")
