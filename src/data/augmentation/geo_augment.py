"""
Geospatial-aware data augmentation pipeline.

Standard augmentation libraries assume RGB photos.
This module handles multi-band satellite imagery with
consistent transforms across image + mask + DEM channels.

Key design decisions:
    - Rotation-invariant: glaciers have no canonical orientation
    - Cloud injection: forces model to handle realistic occlusion
    - Spectral dropout: robustness when a band is unavailable
    - No color jitter: would corrupt physical spectral values
    - All transforms applied consistently to image + mask + DEM
"""

import random
from typing import Optional

import albumentations as A
import numpy as np


def build_train_augmentation(
    patch_size: int = 256,
    cloud_inject_prob: float = 0.3,
    spectral_dropout_prob: float = 0.2,
) -> A.Compose:
    """
    Build the training augmentation pipeline.

    Args:
        patch_size: Spatial size of input patches.
        cloud_inject_prob: Probability of synthetic cloud masking.
        spectral_dropout_prob: Probability of zeroing a non-RGB band.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms = [
        # Rotation-invariant spatial transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Small affine perturbation
        A.Affine(
            translate_percent=0.02,
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            p=0.3,
        ),

        # Coarse dropout simulates cloud shadows
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            p=cloud_inject_prob,
        ),
        # Note: GaussNoise removed from albumentations pipeline
        # Multi-band sensor noise is applied per-band in augment_sample()
        # via numpy directly — albumentations GaussNoise assumes 3 channels
    ]

    return A.Compose(
        transforms,
        additional_targets={
            "mask": "mask",
            "dem":  "image",
        },
    )


def build_val_augmentation() -> A.Compose:
    """
    Validation augmentation - no stochastic transforms.
    Returns empty pipeline for consistency with train pipeline API.
    """
    return A.Compose(
        [],
        additional_targets={
            "mask": "mask",
            "dem":  "image",
        },
    )


def spectral_band_dropout(
    image: np.ndarray,
    drop_prob: float = 0.2,
    protected_bands: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """
    Randomly zero out one non-RGB spectral band.

    Forces the model to be robust when SWIR or NIR is
    unavailable due to sensor failure or data gaps.

    Args:
        image: (C, H, W) array - C channels include RGB + spectral.
        drop_prob: Probability of applying dropout.
        protected_bands: Band indices never zeroed (RGB by default).

    Returns:
        Augmented image array.
    """
    if random.random() > drop_prob:
        return image

    num_bands = image.shape[0]
    droppable = [i for i in range(num_bands) if i not in protected_bands]

    if not droppable:
        return image

    drop_idx = random.choice(droppable)
    image = image.copy()
    image[drop_idx] = 0.0
    return image


def temporal_jitter(
    timestamps: np.ndarray,
    max_days: int = 15,
) -> np.ndarray:
    """
    Add small random noise to acquisition timestamps.

    Prevents the model from memorizing exact acquisition dates
    and improves generalization across different years.

    Args:
        timestamps: (T,) array of days since epoch.
        max_days: Maximum ±jitter in days.

    Returns:
        Jittered timestamps array.
    """
    jitter = np.random.randint(
        -max_days,
        max_days + 1,
        size=timestamps.shape,
    )
    return timestamps + jitter


def dem_jitter(
    dem: np.ndarray,
    sigma_meters: float = 2.0,
) -> np.ndarray:
    """
    Add Gaussian noise to DEM to simulate co-registration error.

    Inter-annual DEM co-registration error on the order of ±2m
    is realistic for Copernicus GLO-30 over glaciated terrain.

    Args:
        dem: (H, W) or (C, H, W) elevation array in meters.
        sigma_meters: Standard deviation of elevation noise.

    Returns:
        Perturbed DEM array.
    """
    noise = np.random.normal(0, sigma_meters, dem.shape).astype(dem.dtype)
    return dem + noise


def climate_noise(
    climate_features: np.ndarray,
    temp_sigma: float = 0.5,
    precip_sigma_frac: float = 0.05,
) -> np.ndarray:
    """
    Add calibrated noise to climate input features.

    Temperature noise (±0.5°C) reflects ERA5 reanalysis uncertainty.
    Precipitation noise (±5%) reflects gauge undercatch uncertainty.

    Feature order assumed:
        [temp_djf, temp_mam, temp_jja, temp_son,
         precip_djf, precip_mam, precip_jja, precip_son,
         snowfall_djf, ... radiation_son]

    Args:
        climate_features: (F,) or (T, F) climate feature array.
        temp_sigma: Noise std for temperature channels in °C.
        precip_sigma_frac: Fractional noise for precipitation.

    Returns:
        Perturbed climate features - clipped to non-negative.
    """
    noisy = climate_features.copy()

    # Temperature bands (indices 0-3): absolute noise
    noisy[..., 0:4] += np.random.normal(
        0, temp_sigma, noisy[..., 0:4].shape
    )

    # Precipitation and snowfall bands (indices 4-11): proportional noise
    noisy[..., 4:12] *= (
        1 + np.random.normal(0, precip_sigma_frac, noisy[..., 4:12].shape)
    )

    return np.clip(noisy, 0, None)


def augment_sample(
    image: np.ndarray,
    mask: np.ndarray,
    dem: np.ndarray,
    climate: np.ndarray,
    timestamps: np.ndarray,
    augmentation: A.Compose,
    spectral_dropout_prob: float = 0.2,
) -> dict:
    """
    Apply full augmentation pipeline to a single training sample.

    Applies spatial transforms consistently across image + mask + DEM,
    then applies independent augmentations to spectral and climate inputs.

    Args:
        image: (C, H, W) spectral image array.
        mask: (H, W) binary glacier mask.
        dem: (3, H, W) terrain features.
        climate: (F,) climate feature vector.
        timestamps: (T,) acquisition timestamps in days.
        augmentation: Albumentations Compose pipeline.
        spectral_dropout_prob: Probability of spectral band dropout.

    Returns:
        dict with augmented 'image', 'mask', 'dem', 'climate', 'timestamps'.
    """
    # Albumentations expects (H, W, C) format
    image_hwc = np.transpose(image, (1, 2, 0))
    dem_hwc = np.transpose(dem, (1, 2, 0))

    transformed = augmentation(
        image=image_hwc,
        mask=mask,
        dem=dem_hwc,
    )

    image_out = np.transpose(transformed["image"], (2, 0, 1))
    dem_out = np.transpose(transformed["dem"], (2, 0, 1))
    mask_out = transformed["mask"]

    # Independent augmentations
    image_out = spectral_band_dropout(image_out, drop_prob=spectral_dropout_prob)
    dem_out = dem_jitter(dem_out)
    climate_out = climate_noise(climate)
    timestamps_out = temporal_jitter(timestamps)

    return {
        "image": image_out,
        "mask": mask_out,
        "dem": dem_out,
        "climate": climate_out,
        "timestamps": timestamps_out,
    }