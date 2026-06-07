"""
Plot Aletsch glacier area over time from Landsat NDSI.
Generates the retreat time series - key paper figure.
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

BBOX           = [7.8, 46.3, 8.2, 46.7]
OUT_DIR        = Path("data/raw/landsat/aletsch")
FIG_DIR        = Path("paper/figures")
PIXEL_AREA_KM2 = 0.0009

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def search_with_retry(date_range, retries=3, wait=15):
    for attempt in range(retries):
        try:
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=BBOX,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": 20}},
                max_items=100,
            )
            items = [
                i for i in search.items()
                if i.datetime.month in (6, 7, 8, 9)
                and i.properties.get("landsat:wrs_path") == "195"
                and i.properties.get("landsat:wrs_row") == "027"
            ]
            return items
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return []


#  Search decade by decade with retry 
print("Searching scenes decade by decade...")
all_items = []

date_ranges = [
    "2000-01-01/2005-12-31",
    "2006-01-01/2011-12-31",
    "2012-01-01/2017-12-31",
    "2018-01-01/2023-12-31",
]

for date_range in date_ranges:
    print(f"  Searching {date_range}...")
    items = search_with_retry(date_range)
    all_items.extend(items)
    print(f"  Found {len(items)} scenes")
    time.sleep(2)

print(f"Total scenes found: {len(all_items)}")

# Pick best scene per year
year_scenes = defaultdict(list)
for item in all_items:
    year_scenes[item.datetime.year].append(item)

best_per_year = {}
for year, scenes in year_scenes.items():
    best_per_year[year] = min(scenes, key=lambda i: i.properties["eo:cloud_cover"])

print(f"Best scenes per year: {sorted(best_per_year.keys())}")

#  Download and compute NDSI per year 
years     = []
areas_km2 = []
ndsi_maps = {}

for year in sorted(best_per_year.keys()):
    item = best_per_year[year]
    print(f"Processing {year}: {item.id} | Cloud: {item.properties['eo:cloud_cover']}%")

    scene_dir = OUT_DIR / item.id
    scene_dir.mkdir(parents=True, exist_ok=True)

    green_path = scene_dir / "green.tif"
    swir1_path = scene_dir / "swir1.tif"

    try:
        download_band(item.assets["green"].href, green_path)
        download_band(item.assets["swir16"].href, swir1_path)
    except Exception as e:
        print(f"  Download failed: {e}")
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

    print(f"  Area: {area:.1f} km²")

if not years:
    print("No data downloaded. Exiting.")
    exit()

#  Plot 1: Retreat time series 
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(years, areas_km2, "o-", color="#2196F3", linewidth=2.5,
        markersize=8, markerfacecolor="white", markeredgewidth=2)
ax.fill_between(years, areas_km2, alpha=0.15, color="#2196F3")

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Glacier Area (km²)", fontsize=13)
ax.set_title("Aletsch Glacier - Area Retreat 2000–2023\n(Landsat NDSI > 0.4 threshold)",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(min(years) - 0.5, max(years) + 0.5)

total_loss = areas_km2[0] - areas_km2[-1]
rate       = total_loss / (years[-1] - years[0])
ax.annotate(
    f"Total loss: {total_loss:.1f} km²\nRate: {rate:.2f} km²/yr",
    xy=(years[-1], areas_km2[-1]),
    xytext=(years[-3], areas_km2[-1] + total_loss * 0.3),
    fontsize=11,
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red",
)

plt.tight_layout()
plt.savefig(FIG_DIR / "aletsch_retreat_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: aletsch_retreat_timeseries.png")

#  Plot 2: NDSI maps over time 
sample_years = [y for y in [2000, 2008, 2016, 2023] if y in ndsi_maps]

if len(sample_years) >= 2:
    fig, axes = plt.subplots(1, len(sample_years), figsize=(5 * len(sample_years), 5))
    if len(sample_years) == 1:
        axes = [axes]

    for ax, year in zip(axes, sample_years):
        im = ax.imshow(ndsi_maps[year], cmap="RdYlBu", vmin=-0.5, vmax=1.0)
        ax.set_title(f"{year}\nArea: {areas_km2[years.index(year)]:.1f} km²",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="NDSI")

    fig.suptitle("Aletsch Glacier - NDSI Maps Over Time",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "aletsch_ndsi_decades.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: aletsch_ndsi_decades.png")

#  Plot 3: Binary glacier mask 
if len(sample_years) >= 2:
    fig, axes = plt.subplots(1, len(sample_years), figsize=(5 * len(sample_years), 5))
    if len(sample_years) == 1:
        axes = [axes]

    for ax, year in zip(axes, sample_years):
        mask = (ndsi_maps[year] > 0.4).astype(np.float32)
        ax.imshow(mask, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{year}\n{areas_km2[years.index(year)]:.1f} km²",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
    
    fig, axes = plt.subplots(1, len(sample_years), figsize=(5 * len(sample_years), 6))

    fig.suptitle("Aletsch Glacier - Binary Glacier Mask Over Time",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG_DIR / "aletsch_mask_decades.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: aletsch_mask_decades.png")