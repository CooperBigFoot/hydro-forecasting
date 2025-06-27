"""
TiDE (Time-series Dense Encoder) model implementation based on the paper:
"Long-term Forecasting with TiDE: Time-series Dense Encoder"
https://arxiv.org/pdf/2304.08424
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import TiDEConfig


class TiDEResBlock(nn.Module):
    """Residual block with optional layer normalization for TiDE model.

    A two-layer MLP with a skip connection and optional LayerNorm.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool = True,
    ):
        """Initialize TiDE residual block.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_size: Size of hidden layer
            dropout: Dropout rate
            use_layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        self.dropout_rate = dropout
        self.skip = nn.Linear(input_dim, output_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor [batch_size, ..., input_dim]

        Returns:
            Output tensor [batch_size, ..., output_dim]
        """
        dense_out = self.dense(x)
        dense_out = functional.dropout(dense_out, p=self.dropout_rate, training=self.training)
        out = dense_out + self.skip(x)
        if self.use_layer_norm:
            out = self.layer_norm(out)
        return out


class TiDEModel(nn.Module):
    """TiDE (Time-series Dense Encoder) model implementation.

    TiDE processes time series data using an encoder-decoder architecture with
    residual blocks. It can handle historical data, future forcing features,
    and static features, fusing them to make predictions.

    The model processes:
      • Historical data (target + past features)
      • Future forcing features (optional)
      • Static features (optional)
    """

    def __init__(self, config: TiDEConfig):
        """Initialize TiDE model.

        Args:
            config: Configuration object for TiDE model
        """
        super().__init__()
        self.config = config

        # Store configuration parameters for convenience
        L = config.input_len
        H = config.output_len
        input_size = config.input_size
        static_size = config.static_size
        future_input_size = config.future_input_size
        output_size = 1  # Output is always a single target variable

        # Calculate past feature dimension (input features minus target variable)
        past_feature_size = max(0, input_size - output_size)

        # Calculate encoder input dimension
        enc_dim = L * output_size  # Target variable features

        # Handle forcing features contribution
        if past_feature_size > 0:
            if config.past_feature_projection_size > 0:
                past_contrib = L * config.past_feature_projection_size
            else:
                past_contrib = L * past_feature_size
            enc_dim += past_contrib

        # Handle future forcing features contribution
        if future_input_size > 0:
            if config.future_forcing_projection_size > 0:
                future_contrib = H * config.future_forcing_projection_size
            else:
                future_contrib = H * future_input_size
            enc_dim += future_contrib

        # Add static features
        if static_size > 0:
            enc_dim += static_size

        # Build the encoder: a stack of residual blocks
        encoder_layers = []
        encoder_layers.append(
            TiDEResBlock(
                enc_dim,
                config.hidden_size,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        )
        for _ in range(config.num_encoder_layers - 1):
            encoder_layers.append(
                TiDEResBlock(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    config.dropout,
                    config.use_layer_norm,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder: a stack of residual blocks
        # Final layer outputs a vector of size = decoder_output_size * H
        decoder_layers = []
        for _ in range(config.num_decoder_layers - 1):
            decoder_layers.append(
                TiDEResBlock(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size,
                    config.dropout,
                    config.use_layer_norm,
                )
            )
        decoder_layers.append(
            TiDEResBlock(
                config.hidden_size,
                config.decoder_output_size * H,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # Temporal decoder: fuses the decoder output with future features
        temporal_in_dim = config.decoder_output_size
        if future_input_size > 0:
            # If a projection is used, the future features are mapped to future_forcing_projection_size
            temporal_in_dim += (
                config.future_forcing_projection_size
                if config.future_forcing_projection_size > 0
                else future_input_size
            )
        self.temporal_decoder = TiDEResBlock(
            temporal_in_dim,
            output_size,  # Always output a single target variable
            config.temporal_decoder_hidden_size,
            config.dropout,
            config.use_layer_norm,
        )

        # Lookback skip connection: projects the past target from length L to H
        self.lookback_skip = nn.Linear(L, H)

        # Optional projections for past and future covariates
        if past_feature_size > 0 and config.past_feature_projection_size > 0:
            self.past_projection = TiDEResBlock(
                past_feature_size,
                config.past_feature_projection_size,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        else:
            self.past_projection = None

        if future_input_size > 0 and config.future_forcing_projection_size > 0:
            self.future_projection = TiDEResBlock(
                future_input_size,
                config.future_forcing_projection_size,
                config.hidden_size,
                config.dropout,
                config.use_layer_norm,
            )
        else:
            self.future_projection = None

        # Store dimensions for convenience in forward pass
        self.output_size = output_size
        self.past_feature_size = past_feature_size

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through TiDE model.

        Args:
            x: Historical data [batch_size, input_len, input_size]
               (contains target as the first feature, followed by optional past features)
            static: Static features [batch_size, static_size] (optional)
            future: Future forcing data [batch_size, output_len, future_input_size] (optional)

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        B, L, _ = x.shape
        H = self.config.output_len
        output_size = self.output_size  # Always 1 for target variable

        # Split x into target and past features
        # Here I assume that the first feature in x is the target variable
        x_target = x[:, :, :output_size]  # [B, L, output_size]

        # Process past features if available
        if self.past_feature_size > 0:
            x_past = x[:, :, output_size : output_size + self.past_feature_size]  # [B, L, past_feature_size]
            if self.past_projection is not None:
                x_past = self.past_projection(x_past)  # [B, L, past_feature_projection_size]
        else:
            x_past = None

        # Process future covariates if provided
        if future is not None and future.shape[-1] > 0:
            future_proj = self.future_projection(future) if self.future_projection is not None else future
        else:
            future_proj = None

        # Build encoder input by flattening and concatenating
        enc_inputs = [x_target.reshape(B, -1)]

        if x_past is not None:
            enc_inputs.append(x_past.reshape(B, -1))
        if future_proj is not None:
            enc_inputs.append(future_proj.reshape(B, -1))
        if static is not None:
            enc_inputs.append(static)

        encoder_input = torch.cat(enc_inputs, dim=1)  # [B, enc_dim]

        # Pass through encoder and decoder
        encoded = self.encoder(encoder_input)
        decoded = self.decoder(encoded)  # [B, decoder_output_size * H]

        # Unflatten the decoder output
        dec_out = decoded.reshape(B, H, -1)  # [B, H, decoder_output_size]

        # Temporal decoding: fuse decoder output with (projected) future forcing if available
        temporal_input = torch.cat([dec_out, future_proj], dim=-1) if future_proj is not None else dec_out
        temporal_decoded = self.temporal_decoder(temporal_input)  # [B, H, output_size]

        # Lookback skip: project the target history from length L to H
        skip = self.lookback_skip(x_target.transpose(1, 2)).transpose(1, 2)  # [B, H, output_size]

        # Final output: add skip connection
        out = temporal_decoded + skip
        return out
