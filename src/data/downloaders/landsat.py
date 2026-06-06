"""
Landsat Collection 2 Level-2 downloader via Microsoft Planetary Computer STAC API.

Downloads surface reflectance scenes for a given glacier bounding box,
date range, and cloud cover threshold. Saves raw GeoTIFF bands to
data/raw/landsat/<glacier_name>/.
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

LANDSAT_BANDS = {
    "blue":  "SR_B2",
    "green": "SR_B3",
    "red":   "SR_B4",
    "nir":   "SR_B5",
    "swir1": "SR_B6",
    "swir2": "SR_B7",
}


def search_landsat(
    bbox: list[float],
    start_date: str,
    end_date: str,
    cloud_cover_max: int = 20,
    max_items: int = 100,
) -> list:
    """
    Search Planetary Computer STAC for Landsat Collection 2 Level-2 scenes.

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
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items=max_items,
        sortby="-datetime",
    )

    items = list(search.items())
    logger.info(f"Found {len(items)} Landsat scenes")
    return items


def download_scene(
    item,
    output_dir: Path,
    bands: Optional[list[str]] = None,
) -> list[Path]:
    """
    Download specific bands from a single Landsat STAC item.

    Args:
        item: STAC item from search_landsat.
        output_dir: Directory to save GeoTIFF files.
        bands: List of band keys from LANDSAT_BANDS. Downloads all if None.

    Returns:
        List of paths to downloaded files.
    """
    if bands is None:
        bands = list(LANDSAT_BANDS.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = item.id
    downloaded = []

    for band_name in bands:
        asset_key = LANDSAT_BANDS[band_name]

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


def download_landsat(
    glacier_name: str,
    bbox: list[float],
    start_date: str,
    end_date: str,
    output_root: Path,
    cloud_cover_max: int = 20,
    season_months: tuple[int, ...] = (6, 7, 8, 9),
    max_scenes: int = 100,
) -> None:
    """
    Full pipeline: search and download Landsat scenes for a glacier.

    Filters to summer months only (ablation season) to ensure
    snow-free glacier imagery for accurate boundary delineation.

    Args:
        glacier_name: Identifier used for output folder name.
        bbox: [lon_min, lat_min, lon_max, lat_max]
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        output_root: Root data directory (e.g. Path("data/raw/landsat"))
        cloud_cover_max: Maximum cloud cover percentage.
        season_months: Months to keep (default: June-September).
        max_scenes: Maximum scenes to download.
    """
    output_dir = output_root / glacier_name
    logger.info(f"Starting Landsat download for {glacier_name}")
    logger.info(f"BBox: {bbox} | Period: {start_date} to {end_date}")

    items = search_landsat(bbox, start_date, end_date, cloud_cover_max, max_scenes)

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
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)