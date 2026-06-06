"""
PyTorch Lightning DataModule for GlacierCastAI.

Handles:
    - Loading pre-extracted .npz patch sequences
    - Train / val / test splitting (temporal - never shuffle across time)
    - Augmentation pipeline (train only)
    - DataLoader creation with num_workers and pin_memory

Temporal split strategy:
    Test set  = last N years (e.g. 2020-2023)
    Val set   = N years before test (e.g. 2017-2019)
    Train set = everything before val

This prevents data leakage - we never train on future observations.
"""

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
    Dataset of temporal glacier sequences.

    Each sample is a dict containing:
        image_seq    : (T, C, H, W) float32 - satellite image sequence
        climate_seq  : (T, F) float32        - climate features per timestep
        dem          : (3, H, W) float32     - terrain features (static)
        target_mask  : (1, H, W) float32     - future glacier boundary
        target_retreat: (3,) float32         - retreat rates 1/3/5yr
        target_risk  : () int64              - risk class 0/1/2
    """

    def __init__(
        self,
        sequences: list[dict],
        augmentation=None,
        is_train: bool = False,
    ):
        """
        Args:
            sequences: Output of build_sequence_index() from patch_extractor.
            augmentation: Albumentations pipeline (train only).
            is_train: Whether to apply augmentation.
        """
        self.sequences = sequences
        self.augmentation = augmentation
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]

        # ── Load input sequence ────────────────────────────────────────
        images, climates, dems = [], [], []

        for path in seq["input_paths"]:
            data = np.load(path)
            images.append(data["image"])    # (C, H, W)
            climates.append(data["climate"] if "climate" in data else np.zeros(16))
            if "dem" in data:
                dems.append(data["dem"])    # (3, H, W)

        image_seq   = np.stack(images, axis=0)    # (T, C, H, W)
        climate_seq = np.stack(climates, axis=0)  # (T, F)
        dem         = dems[0] if dems else np.zeros((3, *image_seq.shape[2:]), dtype=np.float32)

        # ── Load target ────────────────────────────────────────────────
        target_data    = np.load(seq["target_path"])
        target_mask    = target_data["mask"].astype(np.float32)  # (H, W)
        target_mask    = target_mask[None]                        # (1, H, W)
        target_retreat = target_data.get("retreat", np.zeros(3, dtype=np.float32))
        target_risk    = int(target_data.get("risk", 0))

        # ── Augmentation (train only) ──────────────────────────────────
        if self.is_train and self.augmentation is not None:
            # Augment each timestep consistently
            augmented_images = []
            for t in range(image_seq.shape[0]):
                aug = augment_sample(
                    image=image_seq[t],
                    mask=target_mask[0],
                    dem=dem,
                    climate=climate_seq[t],
                    timestamps=np.array([t], dtype=np.float32),
                    augmentation=self.augmentation,
                )
                augmented_images.append(aug["image"])
                dem = aug["dem"]
                climate_seq[t] = aug["climate"]

            image_seq    = np.stack(augmented_images, axis=0)
            target_mask  = aug["mask"][None].astype(np.float32)

        return {
            "image_seq":      torch.from_numpy(image_seq).float(),
            "climate_seq":    torch.from_numpy(climate_seq).float(),
            "dem":            torch.from_numpy(dem).float(),
            "target_mask":    torch.from_numpy(target_mask).float(),
            "target_retreat": torch.from_numpy(
                np.array(target_retreat, dtype=np.float32)
            ),
            "target_risk":    torch.tensor(target_risk, dtype=torch.long),
        }


class GlacierDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for GlacierCastAI.

    Handles all data loading, splitting, and augmentation.
    Compatible with W&B sweep - batch_size is a sweep parameter.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: data section of the full config yaml.
        """
        super().__init__()
        self.config = config

        self.patches_dir  = Path(config.get("patches_dir", "data/processed/patches"))
        self.seq_len      = config["sequences"]["length"]
        self.horizons     = config["sequences"]["horizons"]
        self.batch_size   = config.get("batch_size", 8)
        self.num_workers  = config.get("num_workers", 4)
        self.pin_memory   = config.get("pin_memory", True)
        self.test_years   = config.get("test_years", [2020, 2021, 2022, 2023])
        self.val_years    = config.get("val_years", [2017, 2018, 2019])
        self.glaciers     = list(config["glaciers"].keys())

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build datasets for each split.
        Called automatically by Lightning before training.
        """
        from src.data.preprocessing.patch_extractor import build_sequence_index

        all_sequences = []
        for glacier in self.glaciers:
            for horizon in self.horizons:
                seqs = build_sequence_index(
                    patches_dir=self.patches_dir,
                    glacier_name=glacier,
                    seq_len=self.seq_len,
                    horizon=horizon,
                )
                all_sequences.extend(seqs)

        logger.info(f"Total sequences: {len(all_sequences)}")

        train_seqs, val_seqs, test_seqs = self._temporal_split(all_sequences)

        logger.info(
            f"Split: train={len(train_seqs)} "
            f"val={len(val_seqs)} "
            f"test={len(test_seqs)}"
        )

        train_aug = build_train_augmentation()
        val_aug   = build_val_augmentation()

        self.train_dataset = GlacierSequenceDataset(
            train_seqs, augmentation=train_aug, is_train=True
        )
        self.val_dataset = GlacierSequenceDataset(
            val_seqs, augmentation=val_aug, is_train=False
        )
        self.test_dataset = GlacierSequenceDataset(
            test_seqs, augmentation=val_aug, is_train=False
        )

    def _temporal_split(
        self,
        sequences: list[dict],
    ) -> tuple[list, list, list]:
        """
        Split sequences by year - never shuffle across time.

        Test  = sequences whose target is in test_years
        Val   = sequences whose target is in val_years
        Train = everything else
        """
        train, val, test = [], [], []

        for seq in sequences:
            # Extract year from target filename
            target_stem = Path(seq["target_path"]).stem
            # Filename format: <glacier>_<scene_id>_r<row>_c<col>
            # Scene ID encodes acquisition date e.g. LC08_..._20200912_...
            year = self._extract_year(target_stem)

            if year in self.test_years:
                test.append(seq)
            elif year in self.val_years:
                val.append(seq)
            else:
                train.append(seq)

        return train, val, test

    def _extract_year(self, filename_stem: str) -> int:
        """
        Extract acquisition year from patch filename.
        Falls back to 0 if year cannot be parsed.
        """
        import re
        match = re.search(r"(\d{4})\d{4}", filename_stem)
        if match:
            return int(match.group(1))
        return 0

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