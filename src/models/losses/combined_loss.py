"""
Combined loss functions for GlacierCastAI.

Three loss components:

    1. Segmentation loss (boundary mask):
           Dice + BCE + BoundaryLoss
           Dice handles class imbalance (glacier << background)
           BCE gives pixel-level precision
           BoundaryLoss penalizes coarse edge predictions

    2. Retreat rate loss:
           SmoothL1 — robust to outlier glaciers with extreme retreat

    3. Risk classification loss:
           CrossEntropy with class weights for imbalanced risk tiers

Total loss = seg_loss + λ1 * retreat_loss + λ2 * risk_loss
λ1 and λ2 are hyperparameters included in the W&B sweep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary glacier mask segmentation.

    Handles severe class imbalance: in many patches glacier pixels
    make up < 20% of the total area. BCE alone underperforms here.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits.
            target: (B, 1, H, W) binary mask.
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1).float()

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss: upweights pixels near glacier edges.

    Interior glacier pixels are easy to predict correctly.
    Edge prediction quality directly determines our BF1 metric.
    Higher theta → stronger boundary emphasis.
    """

    def __init__(self, theta: float = 19.0):
        super().__init__()
        self.theta = theta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits.
            target: (B, 1, H, W) binary mask.
        """
        # Morphological boundary: dilation - erosion
        dilated = F.max_pool2d(
            target.float(), kernel_size=3, stride=1, padding=1
        )
        eroded = -F.max_pool2d(
            -target.float(), kernel_size=3, stride=1, padding=1
        )
        boundary = (dilated - eroded).clamp(0, 1)

        # Weight map: boundary pixels get theta weight, rest get 1
        weight = torch.where(
            boundary > 0,
            torch.full_like(boundary, self.theta),
            torch.ones_like(boundary),
        )

        loss = F.binary_cross_entropy_with_logits(
            pred,
            target.float(),
            weight=weight,
        )
        return loss


class SegmentationLoss(nn.Module):
    """
    Combined segmentation loss: Dice + BCE + Boundary.

    Weights are hyperparameters in the W&B sweep.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.3,
        boundary_weight: float = 0.2,
        boundary_theta: float = 19.0,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss(theta=boundary_theta)

        self.dice_w = dice_weight
        self.bce_w = bce_weight
        self.boundary_w = boundary_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.dice_w * self.dice(pred, target)
            + self.bce_w * self.bce(pred, target.float())
            + self.boundary_w * self.boundary(pred, target)
        )


class GlacierForecastLoss(nn.Module):
    """
    Full combined loss for GlacierCastAI.

    Combines all three task losses with configurable weights.
    Individual components are returned for W&B logging so we
    can track which loss dominates during training.

    Loss weights (retreat_weight, risk_weight) are included
    in the hyperparameter sweep — see configs/sweep.yaml.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.3,
        boundary_weight: float = 0.2,
        boundary_theta: float = 19.0,
        retreat_weight: float = 0.5,
        risk_weight: float = 0.3,
        risk_class_weights: torch.Tensor = None,
    ):
        """
        Args:
            dice_weight: Weight for Dice loss component.
            bce_weight: Weight for BCE loss component.
            boundary_weight: Weight for boundary loss component.
            boundary_theta: Boundary emphasis strength.
            retreat_weight: λ1 — weight for retreat rate loss.
            risk_weight: λ2 — weight for risk classification loss.
            risk_class_weights: (3,) tensor for imbalanced risk classes.
        """
        super().__init__()

        self.seg_loss = SegmentationLoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            boundary_weight=boundary_weight,
            boundary_theta=boundary_theta,
        )
        self.smooth_l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss(weight=risk_class_weights)

        self.retreat_w = retreat_weight
        self.risk_w = risk_weight

    def forward(
        self,
        pred_mask: torch.Tensor,
        pred_retreat: torch.Tensor,
        pred_risk: torch.Tensor,
        target_mask: torch.Tensor,
        target_retreat: torch.Tensor,
        target_risk: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_mask: (B, 1, H, W) boundary logits.
            pred_retreat: (B, num_horizons) retreat rate predictions.
            pred_risk: (B, num_classes) risk logits.
            target_mask: (B, 1, H, W) binary glacier mask.
            target_retreat: (B, num_horizons) ground truth retreat rates.
            target_risk: (B,) ground truth risk class indices.

        Returns:
            dict with 'total' and individual loss components for logging.
        """
        seg = self.seg_loss(pred_mask, target_mask)
        retreat = self.smooth_l1(pred_retreat, target_retreat.float())
        risk = self.ce(pred_risk, target_risk.long())

        total = seg + self.retreat_w * retreat + self.risk_w * risk

        return {
            "total":        total,
            "segmentation": seg,
            "retreat":      retreat,
            "risk":         risk,
        }