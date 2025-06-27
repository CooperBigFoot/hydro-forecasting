"""
Implementation of Entity-Aware LSTM (EA-LSTM) for hydrological forecasting.

Based on the paper: "Kratzert et al. (2019) - Towards learning universal, regional, and
local hydrological behaviors via machine learning applied to large-sample datasets"
https://hess.copernicus.org/articles/23/5089/2019/
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import EALSTMConfig


class EALSTMCell(nn.Module):
    """Entity-Aware LSTM cell that modulates input gate using static features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        static_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.static_size = static_size
        self.bias = bias

        # For calculating input gate (i_t) using only static features
        self.weight_sh = nn.Linear(static_size, hidden_size, bias=bias)

        # Dynamic input transformations - use Linear layers instead of Parameter matrices
        self.dynamic_to_gates = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden_to_gates = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Initialize forget gate bias to 1 to help with learning long-term dependencies
        nn.init.constant_(self.dynamic_to_gates.bias.data[:hidden_size], 1.0)

    def forward(
        self,
        dynamic_x: torch.Tensor,  # [batch_size, input_size]
        static_x: torch.Tensor,  # [batch_size, static_size]
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for EA-LSTM cell."""
        batch_size = dynamic_x.size(0)

        # Initialize hidden state if not provided
        if hidden_state is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=dynamic_x.device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=dynamic_x.device)
        else:
            h_0, c_0 = hidden_state

        # Calculate input gate using only static features
        # Important: This is the key difference from standard LSTM
        i_t = torch.sigmoid(self.weight_sh(static_x))

        # Calculate forget, output gates and cell update using dynamic inputs and previous hidden state
        gates = self.dynamic_to_gates(dynamic_x) + self.hidden_to_gates(h_0)

        # Split gates for separate components
        f_t, o_t, g_t = gates.chunk(3, 1)

        # Apply activations and calculate new cell and hidden states
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        c_1 = f_t * c_0 + i_t * g_t
        h_1 = o_t * torch.tanh(c_1)

        return h_1, c_1


class EALSTM(nn.Module):
    """
    Entity-Aware LSTM model for hydrological forecasting.

    This model uses static catchment attributes to modulate the input gate
    of the LSTM, enabling better transfer learning between different catchments.
    """

    def __init__(self, config: EALSTMConfig):
        """
        Initialize the EA-LSTM model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config

        # Create stacked EA-LSTM layers - all layers use the same input_size
        self.ealstm_cells = nn.ModuleList(
            [
                EALSTMCell(
                    input_size=config.input_size,  # Always use config.input_size for all layers
                    hidden_size=config.hidden_size,
                    static_size=config.static_size,
                    bias=config.bias,
                )
                for layer in range(config.num_layers)
            ]
        )

        # Add projection layers between stacked LSTM layers to convert hidden states to input size
        self.hidden_to_input_projections = (
            nn.ModuleList([nn.Linear(config.hidden_size, config.input_size) for _ in range(config.num_layers - 1)])
            if config.num_layers > 1
            else None
        )

        # Add dropout between layers - store dropout rate instead of module
        self.dropout_rate = config.dropout if config.dropout > 0 else 0.0

        # Projection from hidden state to output (for multi-step forecasting)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.output_len),
        )

        # Future forcing integration (if provided)
        if config.future_input_size > 0:
            self.future_forcing_layer = nn.Sequential(
                nn.Linear(config.future_input_size * config.output_len, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.output_len),
            )
        else:
            self.future_forcing_layer = None

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, input_len, input_size]
        static: torch.Tensor,  # [batch_size, static_size]
        future: torch.Tensor | None = None,  # [batch_size, output_len, future_input_size]
        return_hidden: bool = False,  # Flag to control output type
    ) -> torch.Tensor | torch.Tensor:  # [batch_size, output_len, 1] or [batch_size, hidden_size]
        """
        Forward pass of the EA-LSTM model.

        Args:
            x: Dynamic input features [batch_size, input_len, input_size]
               (contains target as the first feature, followed by optional past features)
            static: Static features [batch_size, static_size]
            future: Optional future forcing data [batch_size, output_len, future_input_size]
            return_hidden: Whether to return the final hidden state instead of predictions

        Returns:
            Forecast tensor [batch_size, output_len, 1] if return_hidden is False,
            otherwise the final hidden state [batch_size, hidden_size]
        """
        batch_size = x.size(0)

        # Process each time step through EA-LSTM cells
        hidden_states = [None] * self.config.num_layers

        # Process the sequence through EA-LSTM
        for t in range(self.config.input_len):
            x_t = x[:, t, :]  # [batch_size, input_size]

            # Pass through all LSTM layers
            for layer in range(self.config.num_layers):
                # Get input for this layer
                if layer == 0:
                    layer_input = x_t  # First layer gets raw input
                else:
                    # Get the h_t from the previous layer's output
                    h_t, _ = hidden_states[layer - 1]

                    # Apply dropout if configured - respect training mode
                    if self.dropout_rate > 0:
                        h_t = functional.dropout(h_t, p=self.dropout_rate, training=self.training)

                    # Project hidden state to match expected input size for next layer
                    layer_input = self.hidden_to_input_projections[layer - 1](h_t)

                # Process through EA-LSTM cell
                h_t, c_t = self.ealstm_cells[layer](
                    dynamic_x=layer_input,
                    static_x=static,
                    hidden_state=hidden_states[layer],
                )
                hidden_states[layer] = (h_t, c_t)

        # Use final hidden state for projection
        final_h = hidden_states[-1][0]  # [batch_size, hidden_size]

        # Return hidden state if requested
        if return_hidden:
            return final_h

        # Project hidden state to output sequence
        output = self.projection(final_h)  # [batch_size, output_len]

        # Integrate future forcing if available
        if future is not None and self.future_forcing_layer is not None:
            # Flatten future features
            future_flat = future.reshape(batch_size, -1)  # [batch_size, output_len * future_input_size]

            # Project future features
            future_effect = self.future_forcing_layer(future_flat)  # [batch_size, output_len]

            # Combine with LSTM output
            output = output + future_effect

        # Reshape to [batch_size, output_len, 1]
        return output.unsqueeze(-1)


class BiEALSTM(nn.Module):
    """
    Bidirectional Entity-Aware LSTM model for hydrological forecasting.

    This model uses two separate EA-LSTM branches:
    1. One processes historical data (past)
    2. One processes future forcing data (future)

    The hidden states from both branches are combined to make the final prediction.
    """

    def __init__(self, config: EALSTMConfig):
        """
        Initialize the Bidirectional EA-LSTM model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config

        # EA-LSTM for processing past data
        self.past_ealstm = EALSTM(config)

        future_hidden_size = config.hidden_size  # Always use same hidden size
        future_layers = config.num_layers  # Always use same number of layers

        # Create configuration for the future-processing branch
        future_config = EALSTMConfig(
            input_len=config.output_len,  # Future branch processes the forecast horizon
            output_len=config.output_len,
            input_size=config.future_input_size,
            static_size=config.static_size,
            hidden_size=future_hidden_size,
            num_layers=future_layers,
            bias=config.bias,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
        )

        # EA-LSTM for processing future data
        self.future_ealstm = EALSTM(future_config)

        # Fusion method for combining past and future representations
        self.fusion_method = getattr(config, "bidirectional_fusion", "concat")

        # Combined projection layer
        if self.fusion_method == "concat":
            combined_size = config.hidden_size + future_hidden_size
        elif self.fusion_method in ["add", "average"]:
            # For add/average, dimensions must match
            assert config.hidden_size == future_hidden_size, "Hidden sizes must match for add/average fusion"
            combined_size = config.hidden_size
        else:
            # Default to concatenation
            combined_size = config.hidden_size + future_hidden_size

        # Projection from combined hidden states to output
        self.projection = nn.Sequential(
            nn.Linear(combined_size, combined_size),
            nn.ReLU(),
            nn.Linear(combined_size, config.output_len),
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, input_len, input_size]
        static: torch.Tensor,  # [batch_size, static_size]
        future: torch.Tensor | None = None,  # [batch_size, output_len, future_input_size]
    ) -> torch.Tensor:  # [batch_size, output_len, 1]
        """
        Forward pass of the Bidirectional EA-LSTM model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Forecast tensor [batch_size, output_len, 1]
        """
        # Process past data
        past_hidden = self.past_ealstm(x, static, return_hidden=True)

        if future is None:
            # If no future data provided, fall back to standard EA-LSTM behavior
            output = self.past_ealstm(x, static, future=None)
            return output

        # Process future data
        future_hidden = self.future_ealstm(future, static, return_hidden=True)

        # Combine hidden representations based on fusion method
        if self.fusion_method == "concat":
            combined = torch.cat([past_hidden, future_hidden], dim=1)

        elif self.fusion_method == "add":
            combined = past_hidden + future_hidden
        elif self.fusion_method == "average":
            combined = (past_hidden + future_hidden) / 2
        else:
            # Default to concatenation
            combined = torch.cat([past_hidden, future_hidden], dim=1)

        # Apply nonlinearity for better feature integration
        combined = functional.relu(combined)

        # Project to output sequence
        output = self.projection(combined)  # [batch_size, output_len]

        # Reshape to [batch_size, output_len, 1]
        return output.unsqueeze(-1)
