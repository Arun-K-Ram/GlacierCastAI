"""
Download one Landsat scene for Aletsch glacier and plot it.
Test script to validate the full download → plot pipeline.
"""

import pystac_client
import planetary_computer
import requests
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

#  Config 
BBOX       = [7.8, 46.3, 8.2, 46.7]
OUTPUT_DIR = Path("data/raw/landsat/aletsch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    "red":   "red",
    "green": "green",
    "blue":  "blue",
    "nir":   "nir08",
    "swir1": "swir16",
}

#  Search 
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=BBOX,
    datetime="2020-07-01/2020-09-30",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=5,
    sortby="eo:cloud_cover",
)

items = [i for i in search.items() if i.datetime.month in (6, 7, 8, 9)]
item  = items[0]
print(f"Downloading: {item.id} | {item.datetime.date()} | cloud: {item.properties['eo:cloud_cover']}%")

scene_dir = OUTPUT_DIR / item.id
scene_dir.mkdir(exist_ok=True)

#  Download bands 
downloaded = {}
for band_name, asset_key in BANDS.items():
    if asset_key not in item.assets:
        print(f"  Skipping {asset_key} - not found")
        continue

    url      = item.assets[asset_key].href
    out_path = scene_dir / f"{band_name}.tif"

    if out_path.exists():
        print(f"  Already exists: {out_path.name}")
    else:
        print(f"  Downloading {band_name}...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"  Saved: {out_path.name}")

    downloaded[band_name] = out_path

print(f"\nAll bands downloaded to: {scene_dir}")
print(f"Downloaded bands: {list(downloaded.keys())}")
print(f"Available assets: {list(item.assets.keys())}")

#  Load bands 
bands = {}
for band_name, path in downloaded.items():
    with rasterio.open(path) as src:
        bands[band_name] = src.read(1).astype(np.float32)
        profile = src.profile

#  Normalize helper 
def normalize(arr):
    p2, p98 = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

#  Compute NDSI 
ndsi = (bands["green"] - bands["swir1"]) / (bands["green"] + bands["swir1"] + 1e-8)

#  Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    f"Aletsch Glacier - Landsat\n{item.id}\n{item.datetime.date()}",
    fontsize=14, fontweight="bold"
)

# RGB
rgb = np.stack([
    normalize(bands["red"]),
    normalize(bands["green"]),
    normalize(bands["blue"]),
], axis=-1)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title("RGB True Color")
axes[0, 0].axis("off")

# NIR
axes[0, 1].imshow(normalize(bands["nir"]), cmap="YlGn")
axes[0, 1].set_title("NIR Band")
axes[0, 1].axis("off")

# SWIR1
axes[0, 2].imshow(normalize(bands["swir1"]), cmap="OrRd")
axes[0, 2].set_title("SWIR1 Band")
axes[0, 2].axis("off")

# False color (NIR/Red/Green)
false_color = np.stack([
    normalize(bands["nir"]),
    normalize(bands["red"]),
    normalize(bands["green"]),
], axis=-1)
axes[1, 0].imshow(false_color)
axes[1, 0].set_title("False Color (NIR/R/G)")
axes[1, 0].axis("off")

# NDSI
ndsi_plot = axes[1, 1].imshow(ndsi, cmap="RdYlBu", vmin=-1, vmax=1)
axes[1, 1].set_title("NDSI (Snow/Ice Index)")
axes[1, 1].axis("off")
plt.colorbar(ndsi_plot, ax=axes[1, 1], fraction=0.046)

# Glacier mask from NDSI
glacier_mask = (ndsi > 0.4).astype(np.float32)
axes[1, 2].imshow(rgb)
axes[1, 2].imshow(glacier_mask, cmap="Blues", alpha=0.5, vmin=0, vmax=1)
axes[1, 2].set_title("NDSI Glacier Mask (threshold=0.4)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("paper/figures/aletsch_landsat_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to paper/figures/aletsch_landsat_eda.png")