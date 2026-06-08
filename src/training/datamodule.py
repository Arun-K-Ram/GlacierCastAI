"""
PyTorch Lightning DataModule for GlacierCastAI.

Loads pre-built sequence index from JSON files in
data/processed/sequences/ and serves batches to the trainer.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.data.augmentation.geo_augment import (
    build_train_augmentation,
    build_val_augmentation,
    augment_sample,
)

logger = logging.getLogger(__name__)


class GlacierSequenceDataset(Dataset):
    """
    Dataset that loads temporal glacier sequences from .npz patch files.

    Each sample contains:
        image_seq    : (T, C, H, W) - satellite image sequence
        climate_seq  : (T, F)       - climate features (zeros if unavailable)
        dem          : (3, H, W)    - terrain features
        target_mask  : (1, H, W)   - future glacier boundary
        target_retreat: (3,)        - retreat rates placeholder
        target_risk  : ()           - risk class placeholder
    """

    def __init__(
        self,
        sequences: list,
        augmentation=None,
        is_train: bool = False,
        climate_dim: int = 16,
    ):
        self.sequences   = sequences
        self.augmentation = augmentation
        self.is_train    = is_train
        self.climate_dim = climate_dim

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]

        images = []
        dems   = []

        for path in seq["input_paths"]:
            data = np.load(path)
            img  = data["image"]   # (C, H, W)
            images.append(img)
            if "dem" in data:
                dems.append(data["dem"])

        image_seq = np.stack(images, axis=0)   # (T, C, H, W)
        dem       = dems[0] if dems else np.zeros(
            (3, image_seq.shape[2], image_seq.shape[3]), dtype=np.float32
        )

        # Climate: zeros placeholder until ERA5 is downloaded
        T = image_seq.shape[0]
        climate_seq = np.zeros((T, self.climate_dim), dtype=np.float32)

        # Target
        target_data   = np.load(seq["target_path"])
        target_mask   = target_data["mask"].astype(np.float32)[None]  # (1, H, W)
        target_retreat = np.zeros(3, dtype=np.float32)
        target_risk    = 0

        # Augmentation
        if self.is_train and self.augmentation is not None:
            aug_images = []
            for t in range(T):
                aug = augment_sample(
                    image=image_seq[t],
                    mask=target_mask[0],
                    dem=dem,
                    climate=climate_seq[t],
                    timestamps=np.array([t], dtype=np.float32),
                    augmentation=self.augmentation,
                )
                aug_images.append(aug["image"])
                dem            = aug["dem"]
                climate_seq[t] = aug["climate"]

            image_seq   = np.stack(aug_images, axis=0)
            target_mask = aug["mask"][None].astype(np.float32)

        return {
            "image_seq":      torch.from_numpy(image_seq).float(),
            "climate_seq":    torch.from_numpy(climate_seq).float(),
            "dem":            torch.from_numpy(dem).float(),
            "target_mask":    torch.from_numpy(target_mask).float(),
            "target_retreat": torch.from_numpy(target_retreat).float(),
            "target_risk":    torch.tensor(target_risk, dtype=torch.long),
        }


class GlacierDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for GlacierCastAI.
    Loads sequences from pre-built JSON index files.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        seq_dir          = Path(config.get("sequences_dir", "data/processed/sequences"))
        self.train_json  = seq_dir / "train_sequences.json"
        self.val_json    = seq_dir / "val_sequences.json"
        self.test_json   = seq_dir / "test_sequences.json"

        self.batch_size  = config.get("batch_size", 8)
        self.num_workers = config.get("num_workers", 0)  # 0 for Windows
        self.pin_memory  = config.get("pin_memory", True)
        self.climate_dim = config.get("climate_dim", 16)

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_seqs = json.loads(self.train_json.read_text())
        val_seqs   = json.loads(self.val_json.read_text())
        test_seqs  = json.loads(self.test_json.read_text())

        logger.info(f"Train: {len(train_seqs)} | Val: {len(val_seqs)} | Test: {len(test_seqs)}")

        train_aug = build_train_augmentation()
        val_aug   = build_val_augmentation()

        self.train_dataset = GlacierSequenceDataset(
            train_seqs, augmentation=train_aug,
            is_train=True, climate_dim=self.climate_dim,
        )
        self.val_dataset = GlacierSequenceDataset(
            val_seqs, augmentation=val_aug,
            is_train=False, climate_dim=self.climate_dim,
        )
        self.test_dataset = GlacierSequenceDataset(
            test_seqs, augmentation=val_aug,
            is_train=False, climate_dim=self.climate_dim,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )