"""
Download Copernicus DEM GLO-30 for all 5 study glaciers.
Uses Element84 STAC for search, AWS S3 open data for download.
"""

import time
import pystac_client
import requests
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from pathlib import Path

GLACIERS = {
    "aletsch":   [7.8, 46.3, 8.2, 46.7],
    "gangotri":  [79.0, 30.8, 79.4, 31.1],
    "grey":      [-73.3, -51.0, -72.8, -50.6],
    "columbia":  [-147.2, 61.0, -146.5, 61.4],
    "athabasca": [-117.4, 52.1, -117.1, 52.3],
}

DEM_OUT = Path("data/raw/dem")
FIG_DIR = Path("paper/figures")
DEM_OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

catalog = pystac_client.Client.open(
    "https://earth-search.aws.element84.com/v1",
)


def download_file(url, path):
    if path.exists():
        print(f"    Already exists: {path.name}")
        return True
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    return True


def search_dem_with_retry(bbox, retries=3, wait=15):
    for attempt in range(retries):
        try:
            search = catalog.search(
                collections=["cop-dem-glo-30"],
                bbox=bbox,
            )
            return list(search.items())
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
    return []


def compute_terrain_features(dem):
    """Compute slope and aspect from DEM using numpy gradients."""
    dy, dx      = np.gradient(dem, 30.0)
    slope       = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect      = np.arctan2(-dx, dy)
    aspect_sin  = np.sin(aspect)
    aspect_cos  = np.cos(aspect)
    return slope, aspect_sin, aspect_cos


all_dems = {}

for name, bbox in GLACIERS.items():
    print(f"\n{'='*40}")
    print(f"DEM: {name.upper()}")
    print(f"{'='*40}")

    out_dir = DEM_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    items = search_dem_with_retry(bbox)
    print(f"  Found {len(items)} DEM tiles")

    if not items:
        print(f"  Skipping {name}")
        continue

    tile_paths = []
    for item in items:
        tile_id  = item.id
        # AWS open data public URL - no authentication needed
        aws_url  = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_id}/{tile_id}.tif"
        out_path = out_dir / f"{tile_id}.tif"
        print(f"  Downloading {tile_id}...")
        try:
            download_file(aws_url, out_path)
            tile_paths.append(out_path)
            print(f"  Done: {out_path.name}")
        except Exception as e:
            print(f"  Failed: {e}")

    if not tile_paths:
        print(f"  No tiles downloaded for {name}")
        continue

    # Merge tiles if multiple
    if len(tile_paths) > 1:
        print(f"  Merging {len(tile_paths)} tiles...")
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = merge(datasets)
        profile  = datasets[0].profile.copy()
        profile.update(
            width=mosaic.shape[2],
            height=mosaic.shape[1],
            transform=transform,
        )
        merged_path = out_dir / "dem_merged.tif"
        with rasterio.open(merged_path, "w", **profile) as dst:
            dst.write(mosaic)
        for ds in datasets:
            ds.close()
        dem_path = merged_path
    else:
        dem_path = tile_paths[0]

    # Load and compute terrain features
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem = np.where(dem < -1000, np.nan, dem)

    slope, aspect_sin, aspect_cos = compute_terrain_features(
        np.nan_to_num(dem, nan=0.0)
    )

    all_dems[name] = {
        "dem":        dem,
        "slope":      slope,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
    }

    print(f"  Elevation: {np.nanmin(dem):.0f}m - {np.nanmax(dem):.0f}m")
    print(f"  Slope:     {slope.mean():.1f}° mean")

    # Plot terrain features
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{name.capitalize()} - DEM & Terrain Features",
                 fontsize=13, fontweight="bold")

    im0 = axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("Elevation (m)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(slope, cmap="YlOrRd", vmin=0, vmax=45)
    axes[1].set_title("Slope (°)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(aspect_sin, cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_title("Aspect Sin")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(aspect_cos, cmap="RdBu", vmin=-1, vmax=1)
    axes[3].set_title("Aspect Cos")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_dem.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: paper/figures/{name}_dem.png")

#  Multi-glacier elevation comparison 
if all_dems:
    fig, axes = plt.subplots(1, len(all_dems), figsize=(5 * len(all_dems), 4))
    if len(all_dems) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, all_dems.items()):
        ax.hist(data["dem"].flatten(), bins=50, color="#2196F3",
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name.capitalize()}", fontweight="bold")
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("Pixel count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Elevation Distribution - All Glaciers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "all_glaciers_elevation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: all_glaciers_elevation.png")

print("\nDEM download complete.")