"""
Copernicus DEM GLO-30 downloader via Microsoft Planetary Computer STAC API.

Downloads 30m resolution Digital Elevation Model tiles for a given
glacier bounding box. DEM is static (no time dimension) - downloaded
once per glacier and used as a persistent terrain context channel.

Derived features computed in preprocessing:
    - Slope
    - Aspect (sin and cos components)
    - Topographic Position Index (TPI)
    - Curvature
"""

import logging
from pathlib import Path

import planetary_computer
import pystac_client
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_dem(bbox: list[float]) -> list:
    """
    Search Planetary Computer STAC for Copernicus DEM GLO-30 tiles.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]

    Returns:
        List of STAC items covering the bbox.
    """
    catalog = pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
    )

    items = list(search.items())
    logger.info(f"Found {len(items)} DEM tiles")
    return items


def download_dem(
    glacier_name: str,
    bbox: list[float],
    output_root: Path,
) -> list[Path]:
    """
    Download Copernicus DEM GLO-30 tiles for a glacier region.

    Multiple tiles may be returned if the glacier spans tile boundaries.
    Merging and clipping to the exact bbox is handled in preprocessing.

    Args:
        glacier_name: Identifier used for output folder name.
        bbox: [lon_min, lat_min, lon_max, lat_max]
        output_root: Root directory (e.g. Path("data/raw/dem"))

    Returns:
        List of paths to downloaded DEM tiles.
    """
    output_dir = output_root / glacier_name
    output_dir.mkdir(parents=True, exist_ok=True)

    items = search_dem(bbox)

    if not items:
        logger.error(f"No DEM tiles found for {glacier_name} bbox {bbox}")
        return []

    downloaded = []

    for item in tqdm(items, desc=f"DEM {glacier_name}"):
        if "data" not in item.assets:
            logger.warning(f"No data asset in DEM tile {item.id}")
            continue

        url = item.assets["data"].href
        filename = output_dir / f"{item.id}.tif"

        if filename.exists():
            logger.info(f"Already exists, skipping: {filename.name}")
            downloaded.append(filename)
            continue

        logger.info(f"Downloading DEM tile {item.id}")
        _download_file(url, filename)
        downloaded.append(filename)

    logger.info(f"DEM download complete for {glacier_name} - {len(downloaded)} tiles")
    return downloaded


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream download a file from URL to dest path."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)