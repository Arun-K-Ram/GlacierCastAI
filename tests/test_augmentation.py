"""
Unit tests for geospatial augmentation pipeline.

Run with: poetry run pytest tests/ -v
"""

import numpy as np
import pytest

from src.data.augmentation.geo_augment import (
    spectral_band_dropout,
    temporal_jitter,
    dem_jitter,
    climate_noise,
    build_train_augmentation,
    build_val_augmentation,
)


#  Fixtures 

@pytest.fixture
def sample_image():
    """(C, H, W) multi-spectral image patch."""
    return np.random.rand(7, 256, 256).astype(np.float32)


@pytest.fixture
def sample_mask():
    """(H, W) binary glacier mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 1
    return mask


@pytest.fixture
def sample_dem():
    """(3, H, W) terrain features."""
    return np.random.rand(3, 256, 256).astype(np.float32)


@pytest.fixture
def sample_climate():
    """(16,) climate feature vector."""
    return np.abs(np.random.rand(16).astype(np.float32))


#  Spectral dropout 

def test_spectral_dropout_preserves_rgb(sample_image):
    original_rgb = sample_image[:3].copy()
    augmented = spectral_band_dropout(
        sample_image,
        drop_prob=1.0,
        protected_bands=(0, 1, 2),
    )
    np.testing.assert_array_equal(augmented[:3], original_rgb)


def test_spectral_dropout_zeros_one_band(sample_image):
    augmented = spectral_band_dropout(
        sample_image,
        drop_prob=1.0,
        protected_bands=(0, 1, 2),
    )
    zeroed = [i for i in range(7) if augmented[i].sum() == 0]
    assert len(zeroed) == 1
    assert zeroed[0] not in (0, 1, 2)


def test_spectral_dropout_skips_when_prob_zero(sample_image):
    original = sample_image.copy()
    augmented = spectral_band_dropout(sample_image, drop_prob=0.0)
    np.testing.assert_array_equal(augmented, original)


#  Temporal jitter 

def test_temporal_jitter_shape():
    ts = np.array([0, 365, 730, 1095], dtype=np.float32)
    jittered = temporal_jitter(ts, max_days=15)
    assert jittered.shape == ts.shape


def test_temporal_jitter_within_bounds():
    ts = np.array([0, 365, 730, 1095], dtype=np.float32)
    for _ in range(100):
        jittered = temporal_jitter(ts, max_days=15)
        assert np.all(np.abs(jittered - ts) <= 15)


#  DEM jitter 

def test_dem_jitter_shape(sample_dem):
    jittered = dem_jitter(sample_dem, sigma_meters=2.0)
    assert jittered.shape == sample_dem.shape


def test_dem_jitter_within_5sigma(sample_dem):
    jittered = dem_jitter(sample_dem, sigma_meters=2.0)
    assert np.all(np.abs(jittered - sample_dem) < 20.0)


#  Climate noise 

def test_climate_noise_nonnegative(sample_climate):
    noisy = climate_noise(sample_climate[None])
    assert np.all(noisy >= 0)


def test_climate_noise_shape(sample_climate):
    noisy = climate_noise(sample_climate[None])
    assert noisy.shape == sample_climate[None].shape


#  Augmentation pipeline 

def test_train_augmentation_builds():
    aug = build_train_augmentation()
    assert aug is not None


def test_val_augmentation_builds():
    aug = build_val_augmentation()
    assert aug is not None


def test_train_augmentation_runs(sample_image, sample_mask, sample_dem):
    aug = build_train_augmentation()
    image_hwc = np.transpose(sample_image, (1, 2, 0))
    dem_hwc   = np.transpose(sample_dem, (1, 2, 0))

    result = aug(image=image_hwc, mask=sample_mask, dem=dem_hwc)

    assert result["image"].shape == image_hwc.shape
    assert result["mask"].shape  == sample_mask.shape
    assert result["dem"].shape   == dem_hwc.shape