"""
ERA5 climate data downloader via Copernicus Climate Data Store (CDS) API.

Downloads monthly mean surface variables for a given glacier bounding box
and year range. Variables: 2m temperature, total precipitation, snowfall,
and surface net solar radiation.

ERA5 spatial resolution: 0.25° (~28km) — coarse but the only globally
consistent reanalysis dataset spanning 1940-present.
"""

import logging
from pathlib import Path

import cdsapi
from tqdm import tqdm

logger = logging.getLogger(__name__)

ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "snowfall",
    "surface_net_solar_radiation",
]


def download_era5(
    glacier_name: str,
    bbox: list[float],
    start_year: int,
    end_year: int,
    output_root: Path,
    variables: list[str] = ERA5_VARIABLES,
    months: list[str] = ["06", "07", "08", "09"],
) -> None:
    """
    Download ERA5 monthly means for a glacier region.

    Downloads one NetCDF file per year, containing all requested
    variables and months. Files are saved to:
        output_root/<glacier_name>/era5_<year>.nc

    Args:
        glacier_name: Identifier used for output folder name.
        bbox: [lon_min, lat_min, lon_max, lat_max]
        start_year: First year to download.
        end_year: Last year to download (inclusive).
        output_root: Root directory (e.g. Path("data/raw/climate"))
        variables: ERA5 variable names to download.
        months: Months to download (default: June-September).
    """
    output_dir = output_root / glacier_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ERA5 bbox format: [lat_max, lon_min, lat_min, lon_max]
    # Note: opposite order from STAC bbox
    era5_area = [
        bbox[3],  # lat_max
        bbox[0],  # lon_min
        bbox[1],  # lat_min
        bbox[2],  # lon_max
    ]

    client = cdsapi.Client()

    for year in tqdm(range(start_year, end_year + 1), desc=f"ERA5 {glacier_name}"):
        output_file = output_dir / f"era5_{year}.nc"

        if output_file.exists():
            logger.info(f"Already exists, skipping: {output_file.name}")
            continue

        logger.info(f"Downloading ERA5 {year} for {glacier_name}")

        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": variables,
                "year": str(year),
                "month": months,
                "time": "00:00",
                "area": era5_area,
                "format": "netcdf",
            },
            str(output_file),
        )

        logger.info(f"Saved: {output_file.name}")

    logger.info(f"ERA5 download complete for {glacier_name}")