"""
UNet-style segmentation decoder for glacier boundary prediction.

Takes the fused temporal representation from ConvLSTM or Transformer
and upsamples it back to the original patch resolution (256x256).

Skip connections from the encoder are used to preserve fine-grained
spatial detail - critical for accurate boundary delineation (BF1 metric).

Output: (B, 1, H, W) logit map → sigmoid → glacier probability mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Single UNet decoder block.

    Upsamples by 2x, concatenates skip connection,
    then applies two ConvBNReLU blocks.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        """
        Args:
            in_channels: Channels from upsampled feature map.
            skip_channels: Channels from encoder skip connection.
                           Set to 0 if no skip connection.
            out_channels: Output channels.
        """
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.conv = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) upsampled features.
            skip: (B, skip_channels, 2H, 2W) encoder skip features.

        Returns:
            (B, out_channels, 2H, 2W) decoded features.
        """
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatch from padding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    UNet-style decoder for glacier boundary segmentation.

    Upsamples from encoder bottleneck resolution back to
    full patch resolution (256x256) in 5 steps of 2x each.

    Used for all backbones - skip connections are optional
    (not all backbones expose intermediate feature maps).
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        skip_channels: List[int] = [0, 0, 0, 0, 0],
        num_output_channels: int = 1,
    ):
        """
        Args:
            encoder_dim: Channel dim of encoder output (backbone out_dim).
            decoder_channels: Output channels per decoder block.
            skip_channels: Skip connection channels per block.
                           Set all to 0 if backbone has no skip connections.
            num_output_channels: 1 for binary glacier mask.
        """
        super().__init__()

        assert len(decoder_channels) == len(skip_channels)

        self.blocks = nn.ModuleList()
        in_ch = encoder_dim

        for out_ch, skip_ch in zip(decoder_channels, skip_channels):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # Final 1x1 conv to produce logit map
        self.head = nn.Conv2d(
            decoder_channels[-1],
            num_output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, encoder_dim, H', W') bottleneck feature map.
            skips: Optional list of skip connection tensors
                   from encoder, ordered coarse → fine.

        Returns:
            (B, 1, H, W) logit map (before sigmoid).
        """
        for i, block in enumerate(self.blocks):
            skip = skips[i] if (skips is not None and i < len(skips)) else None
            x = block(x, skip)

        return self.head(x)