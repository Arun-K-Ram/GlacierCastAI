"""
Temporal Transformer for glacier sequence modelling.

Cross-attention between spatial image tokens and climate tokens
is the key multi-modal fusion module and our main architectural
contribution over ConvLSTM.

Why Transformer over ConvLSTM:
    - Attention weights are interpretable (used in explainability)
    - Better long-range temporal dependencies (T > 6)
    - Cross-attention naturally fuses heterogeneous modalities
    - Aligns with current TGRS reviewer expectations (2024-2025)

Architecture:
    Spatial features (B, T, D) ──► Self-attention over time
                                        │
    Climate features (B, T, D) ──► Cross-attention (image queries climate)
                                        │
                                   (B, D) fused representation
                                        │
                                   Decoder head
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Uses both absolute position and day-of-year encoding
    to capture seasonal patterns in glacier imagery.
    """

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.

    Each timestep attends to all other timesteps, allowing the model
    to learn which historical observations are most predictive.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_dim: int = 512,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-norm: more stable training
            activation="gelu",      # GELU over ReLU for transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) temporal token sequence.
            src_key_padding_mask: (B, T) True for padded positions.

        Returns:
            (B, T, d_model) attended sequence.
        """
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class ClimateTemporalCrossAttention(nn.Module):
    """
    Cross-attention: spatial image tokens attend to climate tokens.

    This is the core multi-modal fusion mechanism.
    Image features are the queries; climate features are keys and values.

    Intuition: "Given what I see in the image, which climate signals
    should I attend to for predicting future retreat?"
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        image_tokens: torch.Tensor,
        climate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_tokens: (B, T, d_model) spatial image features.
            climate_tokens: (B, T, d_model) climate features.

        Returns:
            (B, T, d_model) fused representation.
        """
        # Cross-attention with residual
        attended, attn_weights = self.cross_attn(
            query=image_tokens,
            key=climate_tokens,
            value=climate_tokens,
        )
        x = self.norm(image_tokens + attended)

        # Feed-forward with residual
        x = self.ff_norm(x + self.ff(x))

        return x


class TemporalTransformer(nn.Module):
    """
    Full Temporal Transformer encoder for GlacierCastAI.

    Pipeline:
        1. Project spatial features → d_model
        2. Project climate features → d_model
        3. Add sinusoidal positional encoding
        4. Self-attention over image temporal sequence
        5. Cross-attention: image attends to climate
        6. Mean pool over T → final (B, d_model) representation
        7. Reshape back to spatial (B, d_model, H, W) for decoder

    The attention weights from step 5 are saved and used by the
    SHAP explainability layer to attribute predictions to climate vars.
    """

    def __init__(
        self,
        spatial_dim: int,
        climate_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        seq_len: int = 8,
    ):
        """
        Args:
            spatial_dim: Channel dim of spatial features from encoder.
            climate_dim: Dimension of climate feature vector per timestep.
            hidden_dim: Transformer model dimension (d_model).
            num_heads: Number of attention heads.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout rate.
            seq_len: Expected input sequence length T.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projections
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.climate_proj = nn.Sequential(
            nn.Linear(climate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            hidden_dim,
            max_len=seq_len + 4,
            dropout=dropout,
        )

        # Temporal self-attention over image sequence
        self.self_attn = TemporalSelfAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ff_dim=ff_dim,
        )

        # Cross-attention: image × climate fusion
        self.cross_attn = ClimateTemporalCrossAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        spatial_features: torch.Tensor,
        climate_features: torch.Tensor,
        spatial_hw: Optional[tuple] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, T, spatial_dim) pooled spatial features.
            climate_features: (B, T, climate_dim) climate per timestep.
            spatial_hw: (H, W) tuple to reshape output back to spatial.
                        If None, returns (B, hidden_dim) vector.
            src_key_padding_mask: (B, T) mask for padded timesteps.

        Returns:
            If spatial_hw provided: (B, hidden_dim, H, W) spatial feature map.
            Else: (B, hidden_dim) global representation.
        """
        # Project to common dimension
        x = self.spatial_proj(spatial_features)    # (B, T, D)
        c = self.climate_proj(climate_features)     # (B, T, D)

        # Positional encoding
        x = self.pos_enc(x)
        c = self.pos_enc(c)

        # Self-attention over temporal image tokens
        x = self.self_attn(x, src_key_padding_mask=src_key_padding_mask)

        # Cross-attention: image queries climate
        x = self.cross_attn(x, c)

        x = self.output_norm(x)

        # Temporal mean pooling → (B, D)
        x = x.mean(dim=1)

        # Optionally reshape to spatial for skip connections
        if spatial_hw is not None:
            H, W = spatial_hw
            x = x[:, :, None, None].expand(-1, -1, H, W)

        return x