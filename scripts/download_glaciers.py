"""
Download Landsat + ERA5 data for all 5 study glaciers.
Run once to populate data/raw/ for all glaciers.
"""

import time
import pystac_client
import planetary_computer
import requests
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cdsapi
import zipfile
import xarray as xr
from pathlib import Path
from collections import defaultdict

GLACIERS = {
    "aletsch": {
        "bbox":     [7.8, 46.3, 8.2, 46.7],
        "wrs_path": "195",
        "wrs_row":  "027",
        "region":   "Swiss Alps",
    },
    "gangotri": {
        "bbox":     [79.0, 30.8, 79.4, 31.1],
        "wrs_path": "146",
        "wrs_row":  "039",
        "region":   "Himalayas",
    },
    "grey": {
        "bbox":     [-73.3, -51.0, -72.8, -50.6],
        "wrs_path": "232",
        "wrs_row":  "096",
        "region":   "Patagonia",
    },
    "columbia": {
        "bbox":     [-147.2, 61.0, -146.5, 61.4],
        "wrs_path": "069",
        "wrs_row":  "017",
        "region":   "Alaska",
    },
    "athabasca": {
        "bbox":     [-117.4, 52.1, -117.1, 52.3],
        "wrs_path": "044",
        "wrs_row":  "024",
        "region":   "Canada Rockies",
    },
}

LANDSAT_OUT = Path("data/raw/landsat")
CLIMATE_OUT = Path("data/raw/climate")
FIG_DIR     = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_AREA_KM2 = 0.0009

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def download_band(url, path):
    if path.exists():
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def search_with_retry(bbox, wrs_path, wrs_row, date_range, retries=3, wait=15):
    for attempt in range(retries):
        try:
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": 20}},
                max_items=100,
            )
            items = [
                i for i in search.items()
                if i.datetime.month in (6, 7, 8, 9)
                and i.properties.get("landsat:wrs_path") == wrs_path
                and i.properties.get("landsat:wrs_row") == wrs_row
            ]
            return items
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
    return []


def download_landsat_glacier(name, config):
    print(f"\n{'='*50}")
    print(f"LANDSAT: {name.upper()} ({config['region']})")
    print(f"{'='*50}")

    out_dir = LANDSAT_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    date_ranges = [
        "2000-01-01/2005-12-31",
        "2006-01-01/2011-12-31",
        "2012-01-01/2017-12-31",
        "2018-01-01/2023-12-31",
    ]

    all_items = []
    for date_range in date_ranges:
        print(f"  Searching {date_range}...")
        items = search_with_retry(
            config["bbox"],
            config["wrs_path"],
            config["wrs_row"],
            date_range,
        )
        all_items.extend(items)
        print(f"  Found {len(items)} scenes")
        time.sleep(2)

    if not all_items:
        print(f"  No scenes found for {name}")
        return {}, [], []

    # Best scene per year
    year_scenes = defaultdict(list)
    for item in all_items:
        year_scenes[item.datetime.year].append(item)

    best_per_year = {
        year: min(scenes, key=lambda i: i.properties["eo:cloud_cover"])
        for year, scenes in year_scenes.items()
    }

    print(f"  Years found: {sorted(best_per_year.keys())}")

    years     = []
    areas_km2 = []
    ndsi_maps = {}

    for year in sorted(best_per_year.keys()):
        item      = best_per_year[year]
        scene_dir = out_dir / item.id
        scene_dir.mkdir(parents=True, exist_ok=True)

        green_path = scene_dir / "green.tif"
        swir1_path = scene_dir / "swir1.tif"

        try:
            download_band(item.assets["green"].href, green_path)
            download_band(item.assets["swir16"].href, swir1_path)
        except Exception as e:
            print(f"  {year} download failed: {e}")
            continue

        with rasterio.open(green_path) as src:
            green = src.read(1).astype(np.float32)
        with rasterio.open(swir1_path) as src:
            swir1 = src.read(1).astype(np.float32)

        ndsi         = (green - swir1) / (green + swir1 + 1e-8)
        glacier_mask = (ndsi > 0.4)
        area         = glacier_mask.sum() * PIXEL_AREA_KM2

        years.append(year)
        areas_km2.append(area)
        ndsi_maps[year] = ndsi

        print(f"  {year}: {item.id} | cloud={item.properties['eo:cloud_cover']}% | area={area:.1f}km²")

    return ndsi_maps, years, areas_km2


def download_era5_glacier(name, config):
    print(f"\n  ERA5: {name}")

    out_dir = CLIMATE_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "era5_2000_2023.nc"

    if not out_file.exists():
        bbox = config["bbox"]
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type":    "monthly_averaged_reanalysis",
                "variable":        ["2m_temperature", "total_precipitation", "snowfall"],
                "year":            [str(y) for y in range(2000, 2024)],
                "month":           ["06", "07", "08", "09"],
                "time":            "00:00",
                "area":            [bbox[3], bbox[0], bbox[1], bbox[2]],
                "data_format":     "netcdf",
                "download_format": "unarchived",
            },
            str(out_file),
        )
        print(f"  ERA5 saved: {out_file}")
    else:
        print(f"  ERA5 already exists: {out_file.name}")

    # Handle zip
    with open(out_file, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK":
        with zipfile.ZipFile(out_file, "r") as z:
            z.extractall(out_dir)

    nc_files = [f for f in out_dir.glob("*.nc") if "stream" in f.name]
    if not nc_files:
        nc_files = [out_file]

    ds    = xr.open_mfdataset(nc_files, engine="netcdf4", combine="by_coords")
    t2m   = ds["t2m"].mean(dim=["latitude", "longitude"]) - 273.15
    snow  = ds["sf"].mean(dim=["latitude", "longitude"]) * 1000

    years     = sorted(set(t2m.valid_time.dt.year.values))
    temp_ann  = [float(t2m.sel(valid_time=t2m.valid_time.dt.year == y).mean()) for y in years]
    snow_ann  = [float(snow.sel(valid_time=snow.valid_time.dt.year == y).mean()) for y in years]

    return years, temp_ann, snow_ann


def plot_glacier_summary(name, config, ndsi_maps, land_years, areas_km2,
                         clim_years, temp_ann, snow_ann):
    """4-panel summary plot per glacier."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{name.capitalize()} Glacier — Data Summary\n({config['region']})",
        fontsize=16, fontweight="bold",
    )

    # Panel 1: Glacier area retreat
    ax = axes[0, 0]
    ax.plot(land_years, areas_km2, "o-", color="#2196F3", linewidth=2.5,
            markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(land_years, areas_km2, alpha=0.15, color="#2196F3")
    ax.set_title("Glacier Area Over Time (NDSI)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Temperature trend
    ax = axes[0, 1]
    ax.plot(clim_years, temp_ann, "o-", color="#E53935", linewidth=2.5,
            markersize=6, markerfacecolor="white", markeredgewidth=2)
    z = np.polyfit(clim_years, temp_ann, 1)
    ax.plot(clim_years, np.poly1d(z)(clim_years), "--",
            color="#B71C1C", linewidth=1.5, label=f"Trend: {z[0]:+.3f}°C/yr")
    ax.set_title("Summer Temperature Trend (ERA5)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Snowfall trend
    ax = axes[1, 0]
    ax.bar(clim_years, snow_ann, color="#1565C0", alpha=0.7)
    z2 = np.polyfit(clim_years, snow_ann, 1)
    ax.plot(clim_years, np.poly1d(z2)(clim_years), "--",
            color="#0D47A1", linewidth=2, label=f"Trend: {z2[0]:+.4f}mm/yr")
    ax.set_title("Summer Snowfall Trend (ERA5)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Snowfall (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: NDSI map (most recent year)
    ax = axes[1, 1]
    if ndsi_maps:
        latest_year = max(ndsi_maps.keys())
        im = ax.imshow(ndsi_maps[latest_year], cmap="RdYlBu", vmin=-0.5, vmax=1.0)
        ax.set_title(f"NDSI Map — {latest_year}", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="NDSI")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIG_DIR / f"{name}_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


#  Main loop 
all_results = {}

for name, config in GLACIERS.items():
    print(f"\n{'#'*60}")
    print(f"# PROCESSING: {name.upper()}")
    print(f"{'#'*60}")

    ndsi_maps, land_years, areas_km2 = download_landsat_glacier(name, config)
    clim_years, temp_ann, snow_ann   = download_era5_glacier(name, config)

    all_results[name] = {
        "land_years": land_years,
        "areas_km2":  areas_km2,
        "clim_years": clim_years,
        "temp_ann":   temp_ann,
        "snow_ann":   snow_ann,
    }

    if land_years:
        plot_glacier_summary(
            name, config, ndsi_maps,
            land_years, areas_km2,
            clim_years, temp_ann, snow_ann,
        )

#  Multi-glacier comparison plot 
print("\nGenerating multi-glacier comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("All Glaciers — Area and Temperature Trends",
             fontsize=14, fontweight="bold")

colors = ["#2196F3", "#E53935", "#4CAF50", "#FF9800", "#9C27B0"]

for i, (name, res) in enumerate(all_results.items()):
    if not res["land_years"]:
        continue
    axes[0].plot(res["land_years"], res["areas_km2"], "o-",
                 color=colors[i], linewidth=2, label=name.capitalize(),
                 markersize=5)
    axes[1].plot(res["clim_years"], res["temp_ann"], "o-",
                 color=colors[i], linewidth=2, label=name.capitalize(),
                 markersize=5)

axes[0].set_title("Glacier Area Retreat", fontweight="bold")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Area (km²)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Summer Temperature", fontweight="bold")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "all_glaciers_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: all_glaciers_comparison.png")
print("\nAll done.")