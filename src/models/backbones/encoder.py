"""
Spatial encoder backbones for GlacierCastAI.

All backbones are selectable via config string and wrapped in a
common interface: forward() returns a feature map (B, C, H/32, W/32).

Backbones available:
    Classic CNNs  : alexnet, vgg16
    Modern CNNs   : resnet50, resnet101, efficientnet_b4, convnext_base
    Transformers  : vit_b16, swin_t, swin_b, maxvit_t
    Geo Foundation: prithvi_100m

All timm backbones are pretrained on ImageNet-22k by default.
Prithvi-100M is pretrained on NASA HLS satellite data (Landsat + Sentinel-2).

Used in ablation study (Table 2 of paper):
    AlexNet → VGG16 → ResNet50 → ResNet101 → EfficientNet-B4
    → ConvNeXt-B → Swin-T → Swin-B → ViT-B/16 → MaxViT-T → Prithvi-100M
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# Maps config string → timm model name
TIMM_BACKBONE_MAP = {
    "alexnet":          "alexnet",
    "vgg16":            "vgg16",
    "resnet50":         "resnet50",
    "resnet101":        "resnet101",
    "efficientnet_b4":  "efficientnet_b4",
    "convnext_base":    "convnext_base",
    "vit_b16":          "vit_base_patch16_224",
    "swin_t":           "swin_tiny_patch4_window7_224",
    "swin_b":           "swin_base_patch4_window7_224",
    "maxvit_t":         "maxvit_tiny_tf_224",
}

# Output feature dimensions for each backbone
BACKBONE_OUT_DIM = {
    "alexnet":          256,
    "vgg16":            512,
    "resnet50":         2048,
    "resnet101":        2048,
    "efficientnet_b4":  1792,
    "convnext_base":    1024,
    "vit_b16":          768,
    "swin_t":           768,
    "swin_b":           1024,
    "maxvit_t":         512,
    "prithvi_100m":     768,
}


class TimmEncoder(nn.Module):
    """
    Wrapper around timm backbones with a common forward interface.

    Extracts features from the final spatial feature map.
    Supports partial freezing for fine-tuning experiments.
    """

    def __init__(
        self,
        backbone_name: str,
        in_channels: int = 7,
        pretrained: bool = True,
        freeze_epochs: int = 0,
    ):
        """
        Args:
            backbone_name: Key from TIMM_BACKBONE_MAP.
            in_channels: Number of input channels (default 7:
                         RGB + NIR + SWIR1 + SWIR2 + DEM).
            pretrained: Load ImageNet pretrained weights.
            freeze_epochs: Freeze backbone for first N epochs.
        """
        super().__init__()

        if backbone_name not in TIMM_BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Choose from: {list(TIMM_BACKBONE_MAP.keys())}"
            )

        timm_name = TIMM_BACKBONE_MAP[backbone_name]
        self.backbone_name = backbone_name
        self.out_dim = BACKBONE_OUT_DIM[backbone_name]
        self.freeze_epochs = freeze_epochs

        self.model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            global_pool="",         # keep spatial dimensions
            in_chans=in_channels,   # adapt to multi-spectral input
        )

        logger.info(
            f"Loaded backbone: {backbone_name} "
            f"(pretrained={pretrained}, in_channels={in_channels})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, out_dim, H', W') spatial feature map.
        """
        features = self.model.forward_features(x)

        # ViT and some transformers return (B, N, D) sequence tokens
        # Reshape to spatial (B, D, H', W') for ConvLSTM compatibility
        if features.dim() == 3:
            B, N, D = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, D, H, W)

        return features

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"Backbone {self.backbone_name} frozen")

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info(f"Backbone {self.backbone_name} unfrozen")


class PrithviEncoder(nn.Module):
    """
    Prithvi-100M geospatial foundation model encoder.

    Pretrained by IBM and NASA on Harmonized Landsat Sentinel-2 (HLS)
    data - the exact modality we are using. This is our main backbone
    and primary novelty claim over ImageNet-pretrained alternatives.

    Paper: "Prithvi-100M: A geospatial foundation model"
    HuggingFace: ibm-nasa-geospatial/Prithvi-100M
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_epochs: int = 5,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            pretrained: Load HLS-pretrained weights from HuggingFace.
            freeze_epochs: Freeze encoder for first N epochs.
            cache_dir: Local cache directory for model weights.
        """
        super().__init__()

        self.out_dim = BACKBONE_OUT_DIM["prithvi_100m"]
        self.freeze_epochs = freeze_epochs

        if pretrained:
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    "ibm-nasa-geospatial/Prithvi-100M",
                    cache_dir=str(cache_dir) if cache_dir else None,
                    trust_remote_code=True,
                )
                logger.info("Loaded Prithvi-100M from HuggingFace")
            except Exception as e:
                logger.warning(
                    f"Failed to load Prithvi-100M: {e}. "
                    f"Falling back to random init."
                )
                self._init_fallback()
        else:
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Fallback: simple ViT-B with random init."""
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        logger.warning("Using random-init ViT-B as Prithvi fallback")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, out_dim, H', W') spatial feature map.
        """
        outputs = self.model(x)

        # Handle different output formats
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state  # (B, N, D)
        else:
            features = outputs

        if features.dim() == 3:
            B, N, D = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, D, H, W)

        return features

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True


def build_encoder(config: dict) -> nn.Module:
    """
    Factory function - builds encoder from config dict.

    Args:
        config: model.backbone section of model.yaml

    Returns:
        Encoder module with .out_dim and .freeze() / .unfreeze() methods.

    Example config:
        backbone:
            type: prithvi_100m
            pretrained: true
            freeze_epochs: 5
            in_channels: 7
    """
    backbone_type = config.get("type", "resnet50")
    pretrained = config.get("pretrained", True)
    freeze_epochs = config.get("freeze_epochs", 0)
    in_channels = config.get("in_channels", 7)

    if backbone_type == "prithvi_100m":
        return PrithviEncoder(
            pretrained=pretrained,
            freeze_epochs=freeze_epochs,
        )
    else:
        return TimmEncoder(
            backbone_name=backbone_type,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_epochs=freeze_epochs,
        )