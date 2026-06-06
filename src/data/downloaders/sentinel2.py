"""
Sentinel-2 L2A downloader via Microsoft Planetary Computer STAC API.

Downloads surface reflectance scenes for a given glacier bounding box,
date range, and cloud cover threshold. Sentinel-2 provides 10m resolution
vs Landsat's 30m - used for recent years (2015-present).
"""

import logging
from pathlib import Path
from typing import Optional

import planetary_computer
import pystac_client
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

SENTINEL2_BANDS = {
    "blue":  "B02",
    "green": "B03",
    "red":   "B04",
    "nir":   "B08",
    "swir1": "B11",
    "swir2": "B12",
    "scl":   "SCL",   # Scene Classification Layer - used for cloud masking
}


def search_sentinel2(
    bbox: list[float],
    start_date: str,
    end_date: str,
    cloud_cover_max: int = 15,
    max_items: int = 100,
) -> list:
    """
    Search Planetary Computer STAC for Sentinel-2 L2A scenes.

    Note: Sentinel-2 has stricter cloud cover threshold (15% vs 20% for
    Landsat) because its SCL band allows finer cloud masking later.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        cloud_cover_max: Maximum cloud cover percentage.
        max_items: Maximum number of scenes to return.

    Returns:
        List of STAC items.
    """
    catalog = pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items=max_items,
        sortby="-datetime",
    )

    items = list(search.items())
    logger.info(f"Found {len(items)} Sentinel-2 scenes")
    return items


def download_scene(
    item,
    output_dir: Path,
    bands: Optional[list[str]] = None,
) -> list[Path]:
    """
    Download specific bands from a single Sentinel-2 STAC item.

    Always downloads SCL band alongside spectral bands - it is
    required for cloud masking in the preprocessing step.

    Args:
        item: STAC item from search_sentinel2.
        output_dir: Directory to save GeoTIFF files.
        bands: List of band keys from SENTINEL2_BANDS. Downloads all if None.

    Returns:
        List of paths to downloaded files.
    """
    if bands is None:
        bands = list(SENTINEL2_BANDS.keys())

    # Always include SCL for cloud masking
    if "scl" not in bands:
        bands = bands + ["scl"]

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = item.id
    downloaded = []

    for band_name in bands:
        asset_key = SENTINEL2_BANDS[band_name]

        if asset_key not in item.assets:
            logger.warning(f"Band {asset_key} not found in scene {scene_id}")
            continue

        url = item.assets[asset_key].href
        filename = output_dir / f"{scene_id}_{band_name}.tif"

        if filename.exists():
            logger.info(f"Already exists, skipping: {filename.name}")
            downloaded.append(filename)
            continue

        logger.info(f"Downloading {band_name} → {filename.name}")
        _download_file(url, filename)
        downloaded.append(filename)

    return downloaded


def download_sentinel2(
    glacier_name: str,
    bbox: list[float],
    start_date: str,
    end_date: str,
    output_root: Path,
    cloud_cover_max: int = 15,
    season_months: tuple[int, ...] = (6, 7, 8, 9),
    max_scenes: int = 100,
) -> None:
    """
    Full pipeline: search and download Sentinel-2 scenes for a glacier.

    Args:
        glacier_name: Identifier used for output folder name.
        bbox: [lon_min, lat_min, lon_max, lat_max]
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        output_root: Root data directory (e.g. Path("data/raw/sentinel2"))
        cloud_cover_max: Maximum cloud cover percentage.
        season_months: Months to keep (default: June-September).
        max_scenes: Maximum scenes to download.
    """
    output_dir = output_root / glacier_name
    logger.info(f"Starting Sentinel-2 download for {glacier_name}")
    logger.info(f"BBox: {bbox} | Period: {start_date} to {end_date}")

    items = search_sentinel2(bbox, start_date, end_date, cloud_cover_max, max_scenes)

    # Filter to ablation season months only
    items = [
        item for item in items
        if item.datetime.month in season_months
    ]
    logger.info(f"After seasonal filter ({season_months}): {len(items)} scenes")

    for item in tqdm(items, desc=f"Downloading {glacier_name}"):
        download_scene(item, output_dir / item.id)

    logger.info(f"Done. Files saved to {output_dir}")


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream download a file from URL to dest path."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)