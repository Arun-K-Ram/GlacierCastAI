"""
Download Landsat data for all 5 glaciers.
Handles hemisphere differences  southern hemisphere glaciers
use Dec-Feb as their ablation season, not Jun-Sep.
"""

import time
import pystac_client
import planetary_computer
import requests
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from collections import defaultdict

GLACIERS = {
    "aletsch": {
        "bbox":     [7.8, 46.3, 8.2, 46.7],
        "wrs_path": "195",
        "wrs_row":  "027",
        "region":   "Swiss Alps",
        "season":   (6, 7, 8, 9),
        "date_ranges": [
            "2000-01-01/2005-12-31",
            "2006-01-01/2011-12-31",
            "2012-01-01/2017-12-31",
            "2018-01-01/2023-12-31",
        ],
    },
    "gangotri": {
        "bbox":     [79.0, 30.8, 79.4, 31.1],
        "wrs_path": "146",
        "wrs_row":  "039",
        "region":   "Himalayas",
        "season":   (6, 7, 8, 9),
        "date_ranges": [
            "2000-01-01/2005-12-31",
            "2006-01-01/2011-12-31",
            "2012-01-01/2017-12-31",
            "2018-01-01/2023-12-31",
        ],
    },
    "grey": {
        "bbox":     [-73.3, -51.0, -72.8, -50.6],
        "wrs_path": "232",
        "wrs_row":  "096",
        "region":   "Patagonia",
        "season":   (12, 1, 2, 3),  # Southern hemisphere summer
        "date_ranges": [
            "2000-01-01/2005-12-31",
            "2006-01-01/2011-12-31",
            "2012-01-01/2017-12-31",
            "2018-01-01/2023-12-31",
        ],
    },
    "columbia": {
        "bbox":     [-147.2, 61.0, -146.5, 61.4],
        "wrs_path": "068",
        "wrs_row":  "017",
        "region":   "Alaska",
        "season":   (6, 7, 8, 9),
        "date_ranges": [
            "2000-01-01/2005-12-31",
            "2006-01-01/2011-12-31",
            "2012-01-01/2017-12-31",
            "2018-01-01/2023-12-31",
        ],
    },
    "athabasca": {
        "bbox":     [-117.4, 52.1, -117.1, 52.3],
        "wrs_path": "044",
        "wrs_row":  "024",
        "region":   "Canada Rockies",
        "season":   (6, 7, 8, 9),
        "date_ranges": [
            "2000-01-01/2005-12-31",
            "2006-01-01/2011-12-31",
            "2012-01-01/2017-12-31",
            "2018-01-01/2023-12-31",
        ],
    },
}

LANDSAT_OUT = Path("data/raw/landsat")
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


def search_with_retry(bbox, wrs_path, wrs_row, date_range,
                      season, retries=3, wait=15):
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
                if i.datetime.month in season
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


def process_glacier(name, config):
    print(f"\n{'='*50}")
    print(f"{name.upper()}  {config['region']}")
    print(f"{'='*50}")

    out_dir = LANDSAT_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    for date_range in config["date_ranges"]:
        print(f"  Searching {date_range}...")
        items = search_with_retry(
            config["bbox"],
            config["wrs_path"],
            config["wrs_row"],
            date_range,
            config["season"],
        )
        all_items.extend(items)
        print(f"  Found {len(items)} scenes")
        time.sleep(2)

    if not all_items:
        print(f"  No scenes found  trying without WRS filter...")
        # Fallback: search without WRS path/row filter
        for date_range in config["date_ranges"]:
            try:
                search = catalog.search(
                    collections=["landsat-c2-l2"],
                    bbox=config["bbox"],
                    datetime=date_range,
                    query={"eo:cloud_cover": {"lt": 20}},
                    max_items=20,
                )
                items = [
                    i for i in search.items()
                    if i.datetime.month in config["season"]
                ]
                all_items.extend(items)
                print(f"  Fallback found {len(items)} scenes for {date_range}")
                time.sleep(2)
            except Exception as e:
                print(f"  Fallback failed: {e}")

    if not all_items:
        print(f"  No scenes found  skipping {name}")
        return [], [], {}

    # Print what WRS paths/rows we actually got
    wrs_found = set(
        (i.properties.get("landsat:wrs_path"), i.properties.get("landsat:wrs_row"))
        for i in all_items
    )
    print(f"  WRS path/rows found: {wrs_found}")

    # Best scene per year
    year_scenes = defaultdict(list)
    for item in all_items:
        year_scenes[item.datetime.year].append(item)

    best_per_year = {
        year: min(scenes, key=lambda i: i.properties["eo:cloud_cover"])
        for year, scenes in year_scenes.items()
    }

    print(f"  Years: {sorted(best_per_year.keys())}")

    years     = []
    areas_km2 = []
    ndsi_maps = {}

    for year in sorted(best_per_year.keys()):
        item      = best_per_year[year]
        scene_dir = out_dir / item.id
        scene_dir.mkdir(parents=True, exist_ok=True)

        try:
            download_band(item.assets["green"].href, scene_dir / "green.tif")
            download_band(item.assets["swir16"].href, scene_dir / "swir1.tif")
        except Exception as e:
            print(f"  {year} failed: {e}")
            continue

        with rasterio.open(scene_dir / "green.tif") as src:
            green = src.read(1).astype(np.float32)
        with rasterio.open(scene_dir / "swir1.tif") as src:
            swir1 = src.read(1).astype(np.float32)

        ndsi  = (green - swir1) / (green + swir1 + 1e-8)
        area  = (ndsi > 0.4).sum() * PIXEL_AREA_KM2

        years.append(year)
        areas_km2.append(area)
        ndsi_maps[year] = ndsi

        print(
            f"  {year}: area={area:.1f}km² "
            f"cloud={item.properties['eo:cloud_cover']}% "
            f"wrs={item.properties.get('landsat:wrs_path')}/"
            f"{item.properties.get('landsat:wrs_row')}"
        )

    # Plot
    if years:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{name.capitalize()} Glacier  {config['region']}",
            fontsize=14, fontweight="bold",
        )

        axes[0].plot(years, areas_km2, "o-", color="#2196F3", linewidth=2.5,
                     markersize=7, markerfacecolor="white", markeredgewidth=2)
        axes[0].fill_between(years, areas_km2, alpha=0.15, color="#2196F3")
        axes[0].set_title("Glacier Area (NDSI)")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Area (km²)")
        axes[0].grid(True, alpha=0.3)

        latest = max(ndsi_maps.keys())
        im = axes[1].imshow(ndsi_maps[latest], cmap="RdYlBu", vmin=-0.5, vmax=1.0)
        axes[1].set_title(f"NDSI Map  {latest}")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, label="NDSI")

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{name}_landsat.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: paper/figures/{name}_landsat.png")

    return years, areas_km2, ndsi_maps


#  Run all glaciers 
all_results = {}

for name, config in GLACIERS.items():
    # Skip aletsch and athabasca  already downloaded
    if name in ("aletsch", "athabasca"):
        print(f"\nSkipping {name}  already downloaded")
        continue

    years, areas, _ = process_glacier(name, config)
    all_results[name] = {"years": years, "areas": areas}

print("\nDone.")