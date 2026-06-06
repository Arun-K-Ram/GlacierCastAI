"""
Main training entry point for GlacierCastAI.

Usage:
    # Train full model
    python scripts/train.py --config configs/model.yaml

    # Train baseline (image-only, resnet50)
    python scripts/train.py --config configs/model.yaml --backbone resnet50

    # Resume from checkpoint
    python scripts/train.py --config configs/model.yaml --resume experiments/checkpoints/best.ckpt

    # Debug run (2 batches)
    python scripts/train.py --config configs/model.yaml --debug

    # W&B sweep (called automatically by sweep agent)
    python scripts/train.py --config configs/model.yaml --lr 1e-4 --batch-size 8
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.models.glaciercastai import GlacierCastAI
from src.training.trainer import GlacierCastAIModule
from src.training.datamodule import GlacierDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GlacierCastAI")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    # W&B sweep overrides
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--temporal", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=None)
    parser.add_argument("--retreat-weight", type=float, default=None)
    parser.add_argument("--risk-weight", type=float, default=None)

    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args) -> dict:
    """
    Apply CLI / W&B sweep parameter overrides to config.
    This is how the sweep agent injects hyperparameters.
    """
    if args.backbone:
        config["model"]["backbone"]["type"] = args.backbone
    if args.temporal:
        config["model"]["temporal"]["type"] = args.temporal
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.hidden_dim:
        config["model"]["temporal"]["hidden_dim"] = args.hidden_dim
    if args.seq_len:
        config["data"]["sequences"]["length"] = args.seq_len
    if args.freeze_epochs:
        config["model"]["backbone"]["freeze_epochs"] = args.freeze_epochs
    if args.retreat_weight:
        config["model"]["loss_weights"]["retreat_rate"] = args.retreat_weight
    if args.risk_weight:
        config["model"]["loss_weights"]["risk_score"] = args.risk_weight

    return config


def build_callbacks(config: dict, debug: bool) -> list:
    """Build Lightning callbacks."""
    training_cfg = config["training"]
    ckpt_cfg = training_cfg["checkpointing"]

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_cfg.get("dirpath", "experiments/checkpoints"),
            filename="{epoch:02d}-{val/iou:.4f}",
            monitor=ckpt_cfg.get("monitor", "val/iou"),
            mode=ckpt_cfg.get("mode", "max"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            save_last=True,
        ),
        EarlyStopping(
            monitor=training_cfg["early_stopping"]["monitor"],
            patience=training_cfg["early_stopping"]["patience"],
            mode=training_cfg["early_stopping"]["mode"],
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if not debug:
        callbacks.append(RichProgressBar())

    return callbacks


def train(config: dict, args):
    pl.seed_everything(config["training"].get("seed", 42), workers=True)

    # ── W&B Logger ─────────────────────────────────────────────────────
    wb_cfg = config.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wb_cfg.get("project", "GlacierCastAI"),
        entity=wb_cfg.get("entity", None),
        config=config,
        tags=wb_cfg.get("tags", []),
        mode="disabled" if args.debug else "online",
    )

    # ── DataModule ─────────────────────────────────────────────────────
    data_config = {**config["data"], "batch_size": config["training"]["batch_size"]}
    datamodule = GlacierDataModule(data_config)

    # ── Model ──────────────────────────────────────────────────────────
    model = GlacierCastAIModule(config)

    logger.info(f"Model parameter counts: {model.model.count_parameters()}")

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["training"].get("precision", "16-mixed"),
        logger=wandb_logger,
        callbacks=build_callbacks(config, args.debug),
        gradient_clip_val=config["training"].get("gradient_clip", 1.0),
        log_every_n_steps=config.get("wandb", {}).get("log_every_n_steps", 10),
        fast_dev_run=args.debug,
        deterministic=False,    # True slows training significantly
    )

    # ── Train ──────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    # ── Test ───────────────────────────────────────────────────────────
    if not args.debug:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)
    train(config, args)