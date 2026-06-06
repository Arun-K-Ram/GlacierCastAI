"""
GlacierCastAI - Full Model.

Wires together all components into a single forward pass:

    Input:
        image_seq   : (B, T, C, H, W)  - satellite image sequence
        climate_seq : (B, T, F)         - climate features per timestep
        dem         : (B, 3, H, W)      - terrain features (static)

    Pipeline:
        1. Encode each timestep spatially via backbone
        2. Pool spatial features → temporal tokens
        3. Temporal model (ConvLSTM or Transformer) fuses time + climate
        4. Decode → boundary mask (UNet)
        5. Global pool → retreat rate + risk score (MLP heads)

    Output:
        mask    : (B, 1, H, W)          - glacier boundary logits
        retreat : (B, num_horizons)     - retreat rate per horizon
        risk    : (B, num_classes)      - risk tier logits
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones.encoder import build_encoder
from src.models.temporal.convlstm import ConvLSTM
from src.models.temporal.transformer import TemporalTransformer
from src.models.heads.decoder import UNetDecoder
from src.models.heads.regression import MultiHorizonHead
from src.models.losses.combined_loss import GlacierForecastLoss

logger = logging.getLogger(__name__)


class GlacierCastAI(nn.Module):
    """
    Full GlacierCastAI forecasting model.

    Supports all backbone + temporal model combinations
    via config dict - no code changes needed to run ablations.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: model section of model.yaml
        """
        super().__init__()

        backbone_cfg  = config["backbone"]
        temporal_cfg  = config["temporal"]
        climate_cfg   = config["climate_encoder"]
        decoder_cfg   = config.get("decoder", {})
        heads_cfg     = config.get("output_heads", {})

        # ── 1. Spatial encoder (backbone) ──────────────────────────────
        self.encoder = build_encoder(backbone_cfg)
        encoder_dim  = self.encoder.out_dim

        # ── 2. DEM projection ──────────────────────────────────────────
        # DEM has 3 channels (slope, aspect_sin, aspect_cos)
        # Project to match backbone output channels
        self.dem_proj = nn.Conv2d(3, encoder_dim, kernel_size=1)

        # ── 3. Temporal model ──────────────────────────────────────────
        temporal_type = temporal_cfg.get("type", "convlstm")
        hidden_dim    = temporal_cfg.get("hidden_dim", 256)
        climate_dim   = climate_cfg.get("input_dim", 16)

        if temporal_type == "convlstm":
            self.temporal = ConvLSTM(
                input_dim=encoder_dim,
                hidden_dim=hidden_dim,
                num_layers=temporal_cfg.get("num_layers", 3),
                kernel_size=temporal_cfg.get("kernel_size", 3),
                dropout=temporal_cfg.get("dropout", 0.1),
                climate_dim=climate_dim,
            )
            temporal_out_dim = hidden_dim

        elif temporal_type == "transformer":
            self.temporal = TemporalTransformer(
                spatial_dim=encoder_dim,
                climate_dim=climate_dim,
                hidden_dim=hidden_dim,
                num_heads=temporal_cfg.get("num_heads", 8),
                ff_dim=temporal_cfg.get("ff_dim", 512),
                dropout=temporal_cfg.get("dropout", 0.1),
                seq_len=temporal_cfg.get("seq_len", 8),
            )
            temporal_out_dim = hidden_dim

        else:
            raise ValueError(f"Unknown temporal model: {temporal_type}")

        self.temporal_type = temporal_type

        # ── 4. Segmentation decoder ────────────────────────────────────
        self.decoder = UNetDecoder(
            encoder_dim=temporal_out_dim,
            decoder_channels=decoder_cfg.get(
                "channels", [256, 128, 64, 32, 16]
            ),
            skip_channels=decoder_cfg.get(
                "skip_channels", [0, 0, 0, 0, 0]
            ),
            num_output_channels=1,
        )

        # ── 5. Regression + classification heads ───────────────────────
        self.heads = MultiHorizonHead(
            input_dim=temporal_out_dim,
            hidden_dim=heads_cfg.get("hidden_dim", 128),
            dropout=heads_cfg.get("dropout", 0.1),
            num_horizons=3,         # 1yr, 3yr, 5yr
            num_risk_classes=3,     # low, medium, high
        )

        # Global average pool for scalar heads
        self.gap = nn.AdaptiveAvgPool2d(1)

        logger.info(
            f"GlacierCastAI initialized: "
            f"backbone={backbone_cfg['type']} "
            f"temporal={temporal_type} "
            f"hidden_dim={hidden_dim}"
        )

    def forward(
        self,
        image_seq: torch.Tensor,
        climate_seq: torch.Tensor,
        dem: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            image_seq:   (B, T, C, H, W) image sequence.
            climate_seq: (B, T, F) climate features per timestep.
            dem:         (B, 3, H, W) terrain features (static).

        Returns:
            dict with keys:
                'mask'    : (B, 1, H, W) boundary logits
                'retreat' : (B, 3) retreat rates for 1/3/5yr horizons
                'risk'    : (B, 3) risk class logits
        """
        B, T, C, H, W = image_seq.shape

        # ── Step 1: Encode each timestep ───────────────────────────────
        # Process all timesteps in one batch for efficiency
        images_flat = image_seq.view(B * T, C, H, W)
        features_flat = self.encoder(images_flat)   # (B*T, D, H', W')

        _, D, Hf, Wf = features_flat.shape
        features = features_flat.view(B, T, D, Hf, Wf)

        # ── Step 2: Fuse DEM into spatial features ─────────────────────
        dem_feat = self.dem_proj(dem)               # (B, D, H, W)
        dem_feat = F.adaptive_avg_pool2d(dem_feat, (Hf, Wf))

        # Add DEM to every timestep (static terrain context)
        features = features + dem_feat.unsqueeze(1)

        # ── Step 3: Temporal modelling ─────────────────────────────────
        if self.temporal_type == "convlstm":
            # ConvLSTM: returns (B, hidden_dim, H', W')
            temporal_feat = self.temporal(features, climate_seq)

        else:
            # Transformer: needs (B, T, D) tokens
            # Pool spatial dims → temporal tokens
            feat_pooled = features.mean(dim=[-2, -1])   # (B, T, D)
            temporal_feat = self.temporal(
                spatial_features=feat_pooled,
                climate_features=climate_seq,
                spatial_hw=(Hf, Wf),
            )                                            # (B, D, H', W')

        # ── Step 4: Decode → boundary mask ────────────────────────────
        mask = self.decoder(temporal_feat)               # (B, 1, H, W)

        # ── Step 5: Global pool → scalar heads ────────────────────────
        global_feat = self.gap(temporal_feat).squeeze(-1).squeeze(-1)
        retreat, risk = self.heads(global_feat)

        return {
            "mask":    mask,
            "retreat": retreat,
            "risk":    risk,
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone - call during warmup epochs."""
        self.encoder.freeze()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone - call after warmup epochs."""
        self.encoder.unfreeze()

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts per component for paper Table."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "encoder":  count(self.encoder),
            "temporal": count(self.temporal),
            "decoder":  count(self.decoder),
            "heads":    count(self.heads),
            "total":    count(self),
        }