"""
Download ERA5 climate data for all 5 study glaciers.
"""

import cdsapi
import zipfile
import xarray as xr
import numpy as np
from pathlib import Path

GLACIERS = {
    "aletsch":   [7.8, 46.3, 8.2, 46.7],
    "gangotri":  [79.0, 30.8, 79.4, 31.1],
    "grey":      [-73.3, -51.0, -72.8, -50.6],
    "columbia":  [-147.2, 61.0, -146.5, 61.4],
    "athabasca": [-117.4, 52.1, -117.1, 52.3],
}

CLIMATE_OUT = Path("data/raw/climate")
client      = cdsapi.Client()


def download_era5_glacier(name: str, bbox: list):
    out_dir  = CLIMATE_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "era5_2000_2023.nc"

    if out_file.exists():
        print(f"  Already exists: {out_file}")
        return

    print(f"  Downloading ERA5 for {name}...")
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type":    "monthly_averaged_reanalysis",
            "variable":        [
                "2m_temperature",
                "total_precipitation",
                "snowfall",
                "surface_net_solar_radiation",
            ],
            "year":            [str(y) for y in range(2000, 2024)],
            "month":           ["01","02","03","04","05","06",
                               "07","08","09","10","11","12"],
            "time":            "00:00",
            "area":            [bbox[3], bbox[0], bbox[1], bbox[2]],
            "data_format":     "netcdf",
            "download_format": "unarchived",
        },
        str(out_file),
    )
    print(f"  Saved: {out_file}")


for name, bbox in GLACIERS.items():
    print(f"\n{'='*40}")
    print(f"ERA5: {name.upper()}")
    print(f"{'='*40}")
    try:
        download_era5_glacier(name, bbox)
    except Exception as e:
        print(f"  Failed: {e}")

print("\nERA5 download complete.")