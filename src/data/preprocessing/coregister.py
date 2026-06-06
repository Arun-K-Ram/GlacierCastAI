"""
GDAL-based co-registration pipeline.

All raster inputs (Landsat, Sentinel-2, DEM) must be aligned to the
same CRS, resolution, and spatial extent before being fed to the model.
This is non-trivial and often glossed over in papers - we do it properly.

Pipeline per glacier:
    1. Reproject everything to glacier UTM zone
    2. Resample all inputs to 30m (Landsat native resolution)
    3. Clip to glacier bounding box
    4. Align Sentinel-2 (10m) and DEM (30m) to Landsat pixel grid
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject

logger = logging.getLogger(__name__)


def reproject_to_utm(
    src_path: Path,
    dst_path: Path,
    epsg: int,
    resolution: float = 30.0,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reproject a raster to a target UTM CRS at a specified resolution.

    Args:
        src_path: Input raster path.
        dst_path: Output path for reprojected raster.
        epsg: Target EPSG code (e.g. 32632 for UTM Zone 32N).
        resolution: Target pixel size in meters.
        resampling: GDAL resampling method.

    Returns:
        Path to reprojected raster.
    """
    target_crs = CRS.from_epsg(epsg)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )

        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            dtype=rasterio.float32,
            nodata=np.nan,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    logger.info(f"Reprojected to EPSG:{epsg} → {dst_path.name}")
    return dst_path


def coregister_to_reference(
    src_path: Path,
    reference_path: Path,
    dst_path: Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reproject and align src_path to exactly match the grid of reference_path.

    Used to align Sentinel-2 and DEM to the Landsat pixel grid.
    After this, all rasters share the same CRS, transform, and shape.

    Args:
        src_path: Input raster to align.
        reference_path: Reference raster defining target grid.
        dst_path: Output path for aligned raster.
        resampling: GDAL resampling method.

    Returns:
        Path to aligned raster.
    """
    with rasterio.open(reference_path) as ref:
        target_crs = ref.crs
        target_transform = ref.transform
        target_height = ref.height
        target_width = ref.width

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=target_transform,
            width=target_width,
            height=target_height,
            dtype=rasterio.float32,
            nodata=np.nan,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    logger.info(f"Co-registered {src_path.name} → {dst_path.name}")
    return dst_path


def merge_dem_tiles(
    tile_paths: list[Path],
    output_path: Path,
) -> Path:
    """
    Merge multiple DEM tiles into a single raster.

    Copernicus DEM is distributed as 1°x1° tiles - glaciers
    often span two or more tiles and must be merged first.

    Args:
        tile_paths: List of DEM tile paths to merge.
        output_path: Output path for merged DEM.

    Returns:
        Path to merged DEM.
    """
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)

    profile = datasets[0].profile.copy()
    profile.update(
        width=mosaic.shape[2],
        height=mosaic.shape[1],
        transform=transform,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    logger.info(f"Merged {len(tile_paths)} DEM tiles → {output_path.name}")
    return output_path


def compute_ndsi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Snow Index: (Green - SWIR1) / (Green + SWIR1).

    Values > 0.4 indicate snow and ice. Used as an auxiliary input
    channel and for initial glacier mask generation.

    Args:
        green: Green band array.
        swir1: SWIR1 band array.

    Returns:
        NDSI array in [-1, 1].
    """
    denom = green + swir1
    denom = np.where(denom == 0, 1e-8, denom)
    return (green - swir1) / denom


def compute_terrain_features(dem: np.ndarray, resolution: float = 30.0) -> dict:
    """
    Compute terrain features from DEM using numpy gradients.

    Replaces richdem (numpy 2.x incompatible) with direct gradient
    computation. Sufficient for slope and aspect at 30m resolution.

    Args:
        dem: (H, W) elevation array in meters.
        resolution: Pixel size in meters.

    Returns:
        dict with keys: slope, aspect_sin, aspect_cos
    """
    dy, dx = np.gradient(dem, resolution)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Aspect in radians - decomposed into sin/cos to avoid
    # the 0°/360° discontinuity problem
    aspect = np.arctan2(-dx, dy)
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)

    return {
        "slope": slope.astype(np.float32),
        "aspect_sin": aspect_sin.astype(np.float32),
        "aspect_cos": aspect_cos.astype(np.float32),
    }