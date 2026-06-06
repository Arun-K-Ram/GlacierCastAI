"""
Regression and classification heads for GlacierCastAI.

Three output heads on top of the fused temporal representation:

    1. BoundaryHead   - already handled by UNetDecoder (segmentation)
    2. RetreatHead    - predicts annual glacier area loss (km²/yr)
    3. RiskHead       - classifies glacier into risk tier (low/med/high)

These heads take the global (B, D) representation from the temporal
model and produce scalar/class outputs used in the paper's Table 1.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RetreatRateHead(nn.Module):
    """
    Regression head for predicting glacier retreat rate.

    Output: scalar - predicted annual area loss in km²/yr.

    This is used to compute RR-RMSE (our custom metric) and forms
    the retreat_rate loss term in GlacierForecastLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_horizons: int = 3,
    ):
        """
        Args:
            input_dim: Dimension of fused temporal representation.
            hidden_dim: MLP hidden dimension.
            dropout: Dropout rate.
            num_horizons: Number of prediction horizons (default 3: 1yr, 3yr, 5yr).
                          Each horizon gets its own output neuron.
        """
        super().__init__()

        self.num_horizons = num_horizons

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) global fused representation.

        Returns:
            (B, num_horizons) predicted retreat rates in km²/yr.
            Order: [1yr, 3yr, 5yr] by default.
        """
        return self.mlp(x)


class RiskScoreHead(nn.Module):
    """
    Classification head for glacier risk tier prediction.

    Output: 3-class logits - low / medium / high risk of accelerated retreat.

    Risk definition (paper Section 3.4):
        Low:    predicted retreat < 0.5 km²/yr
        Medium: 0.5 - 2.0 km²/yr
        High:   > 2.0 km²/yr

    This enables actionable early warning output beyond the boundary mask.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        """
        Args:
            input_dim: Dimension of fused temporal representation.
            hidden_dim: MLP hidden dimension.
            dropout: Dropout rate.
            num_classes: Number of risk tiers (default 3).
        """
        super().__init__()

        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) global fused representation.

        Returns:
            (B, num_classes) class logits (before softmax).
        """
        return self.mlp(x)


class MultiHorizonHead(nn.Module):
    """
    Combined head for all non-segmentation outputs.

    Wraps RetreatRateHead and RiskScoreHead into a single module
    for clean integration with the main GlacierCastAI model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_horizons: int = 3,
        num_risk_classes: int = 3,
    ):
        super().__init__()

        self.retreat_head = RetreatRateHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_horizons=num_horizons,
        )

        self.risk_head = RiskScoreHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_risk_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) global fused representation.

        Returns:
            retreat: (B, num_horizons) retreat rate predictions.
            risk:    (B, num_risk_classes) risk class logits.
        """
        retreat = self.retreat_head(x)
        risk = self.risk_head(x)
        return retreat, risk