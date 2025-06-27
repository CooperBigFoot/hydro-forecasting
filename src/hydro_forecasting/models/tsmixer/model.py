"""
TSMixer model implementation based on the paper:
"TSMixer: An All-MLP Architecture for Time Series Forecasting"
https://arxiv.org/abs/2303.06053
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import TSMixerConfig


class DeterministicDropout(nn.Dropout):
    """Dropout that explicitly respects training mode for deterministic evaluation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout that explicitly checks training mode."""
        return functional.dropout(x, p=self.p, training=self.training)


class TemporalProjection(nn.Module):
    """Projects input sequence from one length to another using linear transformation.

    This implements the temporal projection operation described in the TSMixer paper,
    which maps sequences from length L to length T.
    """

    def __init__(self, input_len: int, output_len: int, hidden_size: int | None = None):
        """Initialize temporal projection layer.

        Args:
            input_len: Length of input sequence
            output_len: Length of output sequence
            hidden_size: Optional hidden dimension for MLP-based projection
        """
        super().__init__()

        if hidden_size is None:
            # Direct linear projection
            self.projection = nn.Linear(input_len, output_len)
        else:
            # MLP-based projection with hidden layer
            self.projection = nn.Sequential(
                nn.Linear(input_len, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_len),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project tensor along temporal dimension.

        Args:
            x: Input tensor [batch_size, input_len, feature_dim]

        Returns:
            Projected tensor [batch_size, output_len, feature_dim]
        """
        # Transpose to apply projection on temporal dimension
        x_t = x.transpose(1, 2)  # [B, F, L]
        projected = self.projection(x_t)  # [B, F, T]
        return projected.transpose(1, 2)  # [B, T, F]


class AlignmentStage(nn.Module):
    """Aligns historical, future, and static features into a unified representation.

    This implements the alignment stage from the TSMixer paper, which transforms
    heterogeneous inputs into a common representation space through separate
    projection branches followed by concatenation.
    """

    def __init__(
        self,
        input_size: int,
        input_len: int,
        output_len: int,
        future_input_size: int,
        hidden_size: int,
        static_size: int,
        dropout: float = 0.1,
    ):
        """Initialize the alignment stage.

        Args:
            input_size: Number of input features
            input_len: Length of input sequence
            output_len: Length of output sequence
            future_input_size: Number of future forcing features
            hidden_size: Size of hidden representation
            static_size: Number of static features
            dropout: Dropout rate
        """
        super().__init__()

        # Historical feature branch: temporal projection + feature mixing
        self.historical_temporal_proj = TemporalProjection(input_len=input_len, output_len=output_len)
        self.historical_feature_mixing = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            DeterministicDropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.historical_dropout_rate = dropout

        # Future feature branch: temporal projection + feature mixing
        self.future_temporal_proj = TemporalProjection(
            input_len=output_len,
            output_len=output_len,  # Identity projection for consistency
        )
        self.future_feature_mixing = nn.Sequential(
            nn.Linear(future_input_size, hidden_size),
            nn.ReLU(),
            DeterministicDropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.future_dropout_rate = dropout

        # Static feature branch: expansion + feature mixing
        if static_size > 0:
            self.static_feature_mixing = nn.Sequential(
                nn.Linear(static_size, hidden_size),
                nn.ReLU(),
                DeterministicDropout(dropout),
                nn.LayerNorm(hidden_size),
            )
            self.output_size = hidden_size * 3  # Concatenation of all three branches
        else:
            self.static_feature_mixing = None
            self.output_size = hidden_size * 2  # Only historical and future

    def forward(
        self,
        historical: torch.Tensor,  # [batch_size, input_len, input_size]
        future: torch.Tensor,  # [batch_size, output_len, future_input_size]
        static: torch.Tensor | None = None,  # [batch_size, static_size]
    ) -> torch.Tensor:
        """Process and align heterogeneous inputs.

        Args:
            historical: Historical features
            future: Future forcing features
            static: Static features

        Returns:
            Aligned representation [batch_size, output_len, output_size]
        """
        historical.size(0)
        output_len = future.size(1)

        # Process historical features
        hist_projected = self.historical_temporal_proj(historical)
        hist_aligned = self.historical_feature_mixing(hist_projected)

        # Process future features
        future_projected = self.future_temporal_proj(future)
        future_aligned = self.future_feature_mixing(future_projected)

        # Process static features if available
        if static is not None and self.static_feature_mixing is not None:
            # Project static features
            static_aligned = self.static_feature_mixing(static)  # [B, hidden_size]

            # Expand static features to match temporal dimension
            static_aligned = static_aligned.unsqueeze(1).expand(-1, output_len, -1)

            # Concatenate all three aligned representations
            return torch.cat([hist_aligned, future_aligned, static_aligned], dim=-1)
        else:
            # Concatenate only historical and future
            return torch.cat([hist_aligned, future_aligned], dim=-1)


class TimeMixing(nn.Module):
    """Time-mixing MLP that processes features across time.

    Applies mixing operations along the temporal dimension to capture
    temporal patterns in the data.
    """

    def __init__(self, seq_len: int, hidden_size: int, dropout: float = 0.1):
        """Initialize time mixing module.

        Args:
            seq_len: Length of sequence to process
            hidden_size: Size of hidden representation
            dropout: Dropout rate
        """
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, seq_len),
            DeterministicDropout(dropout),
        )

        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time mixing.

        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]

        Returns:
            Time-mixed tensor [batch_size, seq_len, feature_dim]
        """
        # Transpose for time dimension mixing
        x_t = x.transpose(1, 2)  # [B, F, T]

        # Apply MLP along time dimension with residual connection
        mixed = x_t + self.time_mlp(x_t)

        # Apply normalization and transpose back
        return self.norm(mixed).transpose(1, 2)  # [B, T, F]


class FeatureMixing(nn.Module):
    """Feature-mixing MLP that processes time steps across features.

    Applies mixing operations along the feature dimension to capture
    cross-feature interactions.
    """

    def __init__(self, feature_dim: int, hidden_size: int, dropout: float = 0.1):
        """Initialize feature mixing module.

        Args:
            feature_dim: Dimension of input features
            hidden_size: Size of hidden representation
            dropout: Dropout rate
        """
        super().__init__()

        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            DeterministicDropout(dropout),
            nn.Linear(hidden_size, feature_dim),
            DeterministicDropout(dropout),
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature mixing.

        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]

        Returns:
            Feature-mixed tensor [batch_size, seq_len, feature_dim]
        """
        # Apply MLP along feature dimension with residual connection
        mixed = x + self.feature_mlp(x)

        # Apply normalization
        return self.norm(mixed)


class ConditionalFeatureMixing(nn.Module):
    """Conditional feature mixing that uses static features to modulate feature interactions.

    Implements the conditional feature-mixing MLP from the TSMixer paper,
    which incorporates static features to guide the mixing process.
    """

    def __init__(
        self,
        feature_dim: int,
        static_size: int,
        static_embedding_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        """Initialize conditional feature mixing module.

        Args:
            feature_dim: Dimension of input features
            static_size: Dimension of static features
            static_embedding_size: Embedding size for static features
            hidden_size: Size of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        # Static feature processing
        self.static_proj = nn.Sequential(
            nn.Linear(static_size, static_embedding_size),
            nn.ReLU(),
            DeterministicDropout(dropout),
        )

        # Conditioning mechanism (gate)
        self.gate_proj = nn.Linear(static_embedding_size, feature_dim)

        # Feature mixing after modulation
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            DeterministicDropout(dropout),
            nn.Linear(hidden_size, feature_dim),
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, feature_dim]
        static: torch.Tensor,  # [batch_size, static_size]
    ) -> torch.Tensor:
        """Apply conditional feature mixing.

        Args:
            x: Dynamic features to be mixed
            static: Static features for conditioning

        Returns:
            Conditionally mixed features
        """
        # Project static features
        static_emb = self.static_proj(static)  # [B, static_embedding_size]

        # Expand static features to match sequence length
        static_expanded = static_emb.unsqueeze(1).expand(-1, x.size(1), -1)

        # Generate modulation gate
        gate = torch.sigmoid(self.gate_proj(static_expanded))

        # Apply modulation
        x_conditioned = x * gate

        # Apply feature mixing with residual connection
        mixed = x_conditioned + self.feature_mlp(x_conditioned)

        # Apply normalization
        return self.norm(mixed)


class MixerLayer(nn.Module):
    """Mixer layer that combines time mixing and conditional feature mixing.

    Implements the mixer layer from the TSMixer paper, which sequentially
    applies time mixing and feature mixing operations.
    """

    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        hidden_size: int,
        static_size: int,
        static_embedding_size: int,
        dropout: float = 0.1,
    ):
        """Initialize mixer layer.

        Args:
            feature_dim: Dimension of input features
            seq_len: Length of sequence
            hidden_size: Size of hidden layers
            static_size: Dimension of static features
            static_embedding_size: Embedding size for static features
            dropout: Dropout rate
        """
        super().__init__()

        # Time mixing component
        self.time_mixing = TimeMixing(seq_len=seq_len, hidden_size=hidden_size, dropout=dropout)

        # Conditional feature mixing component
        self.feature_mixing = ConditionalFeatureMixing(
            feature_dim=feature_dim,
            static_size=static_size,
            static_embedding_size=static_embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, feature_dim]
        static: torch.Tensor | None = None,  # [batch_size, static_size]
    ) -> torch.Tensor:
        """Apply mixer layer operations.

        Args:
            x: Input tensor
            static: Static features for conditioning

        Returns:
            Processed tensor
        """
        # First apply time mixing
        x_time_mixed = self.time_mixing(x)

        # Then apply conditional feature mixing if static features are provided
        if static is not None:
            return self.feature_mixing(x_time_mixed, static)
        else:
            # Fall back to regular feature mixing without conditioning
            return x_time_mixed


class MixingStage(nn.Module):
    """Mixing stage that processes aligned features through multiple mixer layers.

    Implements the mixing stage from the TSMixer paper, which applies a sequence
    of mixer layers to the aligned features from the alignment stage.
    """

    def __init__(self, config: TSMixerConfig):
        """Initialize mixing stage.

        Args:
            config: TSMixer configuration
        """
        super().__init__()

        # Alignment stage to prepare inputs
        self.alignment_stage = AlignmentStage(
            input_size=config.input_size,
            input_len=config.input_len,
            output_len=config.output_len,
            future_input_size=config.future_input_size,
            hidden_size=config.hidden_size,
            static_size=config.static_size,
            dropout=config.dropout,
        )

        # Determine feature dimension for mixer layers based on alignment output
        feature_dim = config.hidden_size * 3 if config.static_size > 0 else config.hidden_size * 2

        # Store the feature dimension for use by other components
        self.feature_dim = feature_dim

        # Stack of mixer layers
        self.mixer_layers = nn.ModuleList(
            [
                MixerLayer(
                    feature_dim=feature_dim,
                    seq_len=config.output_len,
                    hidden_size=config.hidden_size,
                    static_size=config.static_size,
                    static_embedding_size=config.static_embedding_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_mixing_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, input_len, input_size]
        static: torch.Tensor | None = None,  # [batch_size, static_size]
        future: torch.Tensor | None = None,  # [batch_size, output_len, future_input_size]
    ) -> torch.Tensor:
        """Process inputs through alignment and mixing stages.

        Args:
            x: Historical data
            static: Static features
            future: Future forcing data

        Returns:
            Processed features [batch_size, output_len, feature_dim]
        """
        # Handle case where future forcing is not available
        if future is None:
            # If no future forcing is provided, use zeros
            batch_size = x.size(0)
            output_len = getattr(self.alignment_stage, "output_len", x.size(1))
            future_input_size = getattr(self.alignment_stage.future_feature_mixing[0], "in_features", 1)
            future = torch.zeros(batch_size, output_len, future_input_size, device=x.device)

        # Align features
        aligned = self.alignment_stage(historical=x, future=future, static=static)

        # Process through mixer layers
        features = aligned
        for layer in self.mixer_layers:
            features = layer(features, static)

        return features


class TSMixerHead(nn.Module):
    """Output head for TSMixer that projects mixed features to predictions.

    Implements the final prediction component that projects the mixed features
    to the target output.
    """

    def __init__(self, feature_dim: int, hidden_size: int, output_dim: int = 1):
        """Initialize TSMixer head.

        Args:
            feature_dim: Dimension of input features
            hidden_size: Size of hidden layers
            output_dim: Dimension of output predictions
        """
        super().__init__()

        self.prediction = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions from features.

        Args:
            x: Input features [batch_size, seq_len, feature_dim]

        Returns:
            Predictions [batch_size, seq_len, output_dim]
        """
        return self.prediction(x)


class TSMixer(nn.Module):
    """TSMixer model for time series forecasting with auxiliary information.

    Implements the complete TSMixer architecture from the paper, incorporating
    alignment of heterogeneous inputs, mixing through multiple MLP layers, and
    final projection to forecast outputs.
    """

    def __init__(self, config: TSMixerConfig):
        """Initialize TSMixer model.

        Args:
            config: TSMixer configuration
        """
        super().__init__()

        self.config = config

        # Mixing stage implementation
        self.mixing_stage = MixingStage(config)

        # Head using the feature dimension from the mixing stage
        self.head = TSMixerHead(feature_dim=self.mixing_stage.feature_dim, hidden_size=config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate forecasts from input data.

        Args:
            x: Historical input features [B, input_len, input_size]
            static: Static features [B, static_size]
            future: Future forcing data [B, output_len, future_input_size]

        Returns:
            Predictions [B, output_len, 1]
        """
        # Validate input dimensions
        assert x.ndim == 3, "Input tensor x must be of shape [B, input_len, input_size]"
        if static is not None:
            assert static.ndim == 2, "Static tensor must be of shape [B, static_size]"
        if future is not None:
            assert future.ndim == 3, "Future tensor must be of shape [B, output_len, future_input_size]"

        # Process through mixing stage
        features = self.mixing_stage(x, static, future)

        # Generate predictions
        return self.head(features)
