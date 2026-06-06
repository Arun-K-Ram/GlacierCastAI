"""
Cloud masking for Sentinel-2 and Landsat imagery.

Sentinel-2: uses the Scene Classification Layer (SCL band)
Landsat: uses the QA_PIXEL band (CFMask algorithm)

Masked pixels are set to NaN so downstream code can handle
them explicitly rather than treating clouds as valid data.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# Sentinel-2 SCL class values to mask out
# https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
SCL_MASK_CLASSES = {
    0: "no_data",
    1: "saturated_defective",
    2: "dark_area",
    3: "cloud_shadow",
    8: "cloud_medium_prob",
    9: "cloud_high_prob",
    10: "thin_cirrus",
    11: "snow_ice",   # keep this False if you want snow pixels
}

# Landsat QA_PIXEL bit flags to mask
# Bit 1: Dilated cloud
# Bit 3: Cloud
# Bit 4: Cloud shadow
LANDSAT_QA_MASK_BITS = [1, 3, 4]


def apply_scl_mask(
    image: np.ndarray,
    scl: np.ndarray,
    mask_snow: bool = False,
) -> np.ndarray:
    """
    Apply Sentinel-2 SCL cloud mask to a multi-band image array.

    Args:
        image: (C, H, W) float32 array of spectral bands.
        scl: (H, W) uint8 SCL band array.
        mask_snow: If True, also mask snow/ice pixels (SCL class 11).
                   Set False during glacier mapping - snow IS the signal.

    Returns:
        (C, H, W) array with masked pixels set to NaN.
    """
    mask_classes = list(SCL_MASK_CLASSES.keys())
    if not mask_snow:
        mask_classes = [c for c in mask_classes if c != 11]

    cloud_mask = np.isin(scl, mask_classes)  # True where cloudy

    masked = image.astype(np.float32).copy()
    masked[:, cloud_mask] = np.nan

    cloud_pct = cloud_mask.mean() * 100
    logger.debug(f"SCL mask applied - {cloud_pct:.1f}% pixels masked")

    return masked


def apply_qa_mask(
    image: np.ndarray,
    qa_pixel: np.ndarray,
) -> np.ndarray:
    """
    Apply Landsat QA_PIXEL CFMask cloud mask to a multi-band image.

    Args:
        image: (C, H, W) float32 array of spectral bands.
        qa_pixel: (H, W) uint16 QA_PIXEL band array.

    Returns:
        (C, H, W) array with masked pixels set to NaN.
    """
    cloud_mask = np.zeros(qa_pixel.shape, dtype=bool)

    for bit in LANDSAT_QA_MASK_BITS:
        cloud_mask |= ((qa_pixel >> bit) & 1).astype(bool)

    masked = image.astype(np.float32).copy()
    masked[:, cloud_mask] = np.nan

    cloud_pct = cloud_mask.mean() * 100
    logger.debug(f"QA mask applied - {cloud_pct:.1f}% pixels masked")

    return masked


def load_and_mask_sentinel2(
    scene_dir: Path,
    mask_snow: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Load all Sentinel-2 bands from a scene directory and apply SCL mask.

    Expects files named: <scene_id>_<band>.tif
    Band order: blue, green, red, nir, swir1, swir2

    Args:
        scene_dir: Directory containing band GeoTIFFs for one scene.
        mask_snow: Whether to mask snow/ice pixels.

    Returns:
        Tuple of:
            image: (6, H, W) float32 masked array
            meta: dict with rasterio profile and scene metadata
    """
    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
    bands = []
    profile = None

    for band_name in band_names:
        tif_files = list(scene_dir.glob(f"*_{band_name}.tif"))
        if not tif_files:
            raise FileNotFoundError(f"Band {band_name} not found in {scene_dir}")

        with rasterio.open(tif_files[0]) as src:
            bands.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()

    image = np.stack(bands, axis=0)  # (6, H, W)

    # Load SCL
    scl_files = list(scene_dir.glob("*_scl.tif"))
    if scl_files:
        with rasterio.open(scl_files[0]) as src:
            scl = src.read(1)
        image = apply_scl_mask(image, scl, mask_snow=mask_snow)
    else:
        logger.warning(f"No SCL file found in {scene_dir} - skipping cloud mask")

    return image, {"profile": profile, "scene_dir": scene_dir}


def load_and_mask_landsat(
    scene_dir: Path,
) -> tuple[np.ndarray, dict]:
    """
    Load all Landsat bands from a scene directory and apply QA mask.

    Band order: blue, green, red, nir, swir1, swir2

    Args:
        scene_dir: Directory containing band GeoTIFFs for one scene.

    Returns:
        Tuple of:
            image: (6, H, W) float32 masked array
            meta: dict with rasterio profile and scene metadata
    """
    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
    bands = []
    profile = None

    for band_name in band_names:
        tif_files = list(scene_dir.glob(f"*_{band_name}.tif"))
        if not tif_files:
            raise FileNotFoundError(f"Band {band_name} not found in {scene_dir}")

        with rasterio.open(tif_files[0]) as src:
            bands.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()

    image = np.stack(bands, axis=0)  # (6, H, W)

    # Load QA_PIXEL
    qa_files = list(scene_dir.glob("*QA_PIXEL*.tif"))
    if qa_files:
        with rasterio.open(qa_files[0]) as src:
            qa = src.read(1)
        image = apply_qa_mask(image, qa)
    else:
        logger.warning(f"No QA_PIXEL file found in {scene_dir} - skipping cloud mask")

    return image, {"profile": profile, "scene_dir": scene_dir}