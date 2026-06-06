"""
PyTorch Lightning training module for GlacierCastAI.

Handles:
    - Forward pass + loss computation
    - Backpropagation (automatic via Lightning)
    - Optimizer + LR scheduler
    - Backbone freeze/unfreeze schedule
    - W&B metric logging
    - Hyperparameter sweep compatibility

Lightning abstracts the training loop:
    for each batch:
        outputs = model.forward(batch)       ← our code
        loss = loss_fn(outputs, targets)     ← our code
        loss.backward()                      ← Lightning automatic
        optimizer.step()                     ← Lightning automatic
        scheduler.step()                     ← Lightning automatic
"""

import logging
from typing import Dict, Optional

import torch
import pytorch_lightning as pl
import wandb

from src.models.glaciercastai import GlacierCastAI
from src.models.losses.combined_loss import GlacierForecastLoss
from src.evaluation.metrics import compute_iou, compute_boundary_f1

logger = logging.getLogger(__name__)


class GlacierCastAIModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping GlacierCastAI.

    All hyperparameters come from config dict - compatible
    with W&B sweep parameter injection via train.py.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict (model + training sections).
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        model_cfg    = config["model"]
        training_cfg = config["training"]
        loss_cfg     = model_cfg.get("loss_weights", {})

        # ── Model ──────────────────────────────────────────────────────
        self.model = GlacierCastAI(model_cfg)

        # ── Loss ───────────────────────────────────────────────────────
        self.loss_fn = GlacierForecastLoss(
            dice_weight=loss_cfg.get("dice", 0.5),
            bce_weight=loss_cfg.get("bce", 0.3),
            boundary_weight=loss_cfg.get("boundary", 0.2),
            retreat_weight=loss_cfg.get("retreat_rate", 0.5),
            risk_weight=loss_cfg.get("risk_score", 0.3),
        )

        # ── Training config ────────────────────────────────────────────
        self.lr            = training_cfg["optimizer"]["lr"]
        self.weight_decay  = training_cfg["optimizer"]["weight_decay"]
        self.freeze_epochs = model_cfg["backbone"].get("freeze_epochs", 0)

        # Track validation IoU for checkpointing
        self.best_val_iou = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        image_seq: torch.Tensor,
        climate_seq: torch.Tensor,
        dem: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.model(image_seq, climate_seq, dem)

    # ──────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch)
        losses  = self._compute_loss(outputs, batch)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return losses["total"]

    # ──────────────────────────────────────────────────────────────────
    # Validation step
    # ──────────────────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self._shared_step(batch)
        losses  = self._compute_loss(outputs, batch)

        # Compute evaluation metrics
        metrics = self._compute_metrics(outputs, batch)

        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_epoch=True,
            prog_bar=False,
        )
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_epoch=True,
            prog_bar=True,
        )

    # ──────────────────────────────────────────────────────────────────
    # Test step
    # ──────────────────────────────────────────────────────────────────

    def test_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self._shared_step(batch)
        metrics = self._compute_metrics(outputs, batch)

        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_epoch=True,
        )

    # ──────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────

    def _shared_step(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Run forward pass on a batch."""
        return self.model(
            image_seq=batch["image_seq"],
            climate_seq=batch["climate_seq"],
            dem=batch["dem"],
        )

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: dict,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        return self.loss_fn(
            pred_mask=outputs["mask"],
            pred_retreat=outputs["retreat"],
            pred_risk=outputs["risk"],
            target_mask=batch["target_mask"],
            target_retreat=batch["target_retreat"],
            target_risk=batch["target_risk"],
        )

    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: dict,
    ) -> Dict[str, float]:
        """Compute IoU and BF1 on CPU numpy arrays."""
        import numpy as np

        pred_mask = torch.sigmoid(outputs["mask"]).squeeze(1)  # (B, H, W)
        target_mask = batch["target_mask"].squeeze(1)           # (B, H, W)

        ious, bf1s = [], []

        for i in range(pred_mask.shape[0]):
            pred_np   = pred_mask[i].detach().cpu().numpy()
            target_np = target_mask[i].detach().cpu().numpy()

            ious.append(compute_iou(pred_np, target_np))
            bf1s.append(compute_boundary_f1(pred_np, target_np)["f1"])

        return {
            "iou":         float(np.mean(ious)),
            "boundary_f1": float(np.mean(bf1s)),
        }

    # ──────────────────────────────────────────────────────────────────
    # Optimizer + scheduler
    # ──────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        AdamW optimizer with cosine annealing + linear warmup.

        Both optimizer and scheduler hyperparameters are included
        in the W&B sweep search space (configs/sweep.yaml).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        training_cfg = self.config["training"]
        warmup_epochs = training_cfg["scheduler"].get("warmup_epochs", 10)
        total_epochs  = training_cfg.get("epochs", 100)
        min_lr        = training_cfg["scheduler"].get("min_lr", 1e-6)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch + 1) / float(warmup_epochs)
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                return max(
                    min_lr / self.lr,
                    0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()),
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/iou",
            },
        }

    # ──────────────────────────────────────────────────────────────────
    # Backbone freeze schedule
    # ──────────────────────────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        """
        Freeze backbone for first N epochs, then unfreeze.

        Prevents large pretrained weights from overwriting
        randomly initialized decoder weights early in training.
        freeze_epochs is a hyperparameter in the W&B sweep.
        """
        if self.current_epoch == 0 and self.freeze_epochs > 0:
            self.model.freeze_backbone()
            logger.info(f"Backbone frozen for {self.freeze_epochs} epochs")

        if self.current_epoch == self.freeze_epochs and self.freeze_epochs > 0:
            self.model.unfreeze_backbone()
            logger.info(f"Backbone unfrozen at epoch {self.current_epoch}")

    # ──────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────

    def on_validation_epoch_end(self) -> None:
        """Log parameter counts once to W&B."""
        if self.current_epoch == 0:
            param_counts = self.model.count_parameters()
            if wandb.run is not None:
                wandb.run.summary.update(param_counts)
            logger.info(f"Parameter counts: {param_counts}")