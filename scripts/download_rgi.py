"""
Create approximate glacier boundary polygons from bounding boxes.

For paper-quality results, the RGI polygons should be downloaded from
NSIDC (requires Earthdata login) and placed in data/raw/rgi/.

For now we create bbox polygons as placeholders - sufficient for
testing the preprocessing pipeline. Replace with RGI shapefiles
when available.

RGI 6.0 download: https://nsidc.org/data/nsidc-0770/versions/6
Earthdata login:  https://urs.earthdata.nasa.gov
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from shapely.geometry import box

RGI_DIR = Path("data/raw/rgi")
FIG_DIR = Path("paper/figures")
RGI_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Known approximate glacier areas from literature (km²)
# Used as ground truth for RR-RMSE metric
GLACIERS = {
    "aletsch": {
        "bbox":        [7.93, 46.38, 8.13, 46.62],  # Tighter bbox around glacier
        "area_km2":    81.7,   # 2020 estimate from literature
        "region":      "Swiss Alps",
        "rgi_id":      "RGI60-11.01450",
        "crs":         "EPSG:4326",
    },
    "gangotri": {
        "bbox":        [79.05, 30.85, 79.35, 31.05],
        "area_km2":    143.0,
        "region":      "Himalayas",
        "rgi_id":      "RGI60-14.04477",
        "crs":         "EPSG:4326",
    },
    "grey": {
        "bbox":        [-73.25, -50.95, -72.90, -50.65],
        "area_km2":    270.0,
        "region":      "Patagonia",
        "rgi_id":      "RGI60-17.00353",
        "crs":         "EPSG:4326",
    },
    "columbia": {
        "bbox":        [-147.15, 61.05, -146.60, 61.35],
        "area_km2":    900.0,
        "region":      "Alaska",
        "rgi_id":      "RGI60-01.10689",
        "crs":         "EPSG:4326",
    },
    "athabasca": {
        "bbox":        [-117.35, 52.15, -117.15, 52.28],
        "area_km2":    17.8,
        "region":      "Canada Rockies",
        "rgi_id":      "RGI60-02.18778",
        "crs":         "EPSG:4326",
    },
}

print("Creating glacier boundary polygons...")
glacier_gdfs = {}

for name, info in GLACIERS.items():
    bbox    = info["bbox"]
    polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])

    gdf = gpd.GeoDataFrame(
        {
            "glacier_name": [name],
            "rgi_id":       [info["rgi_id"]],
            "area_km2":     [info["area_km2"]],
            "region":       [info["region"]],
            "source":       ["bbox_placeholder"],
        },
        geometry=[polygon],
        crs=info["crs"],
    )

    out_path = RGI_DIR / f"{name}.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    glacier_gdfs[name] = gdf
    print(f"  {name}: {out_path.name} | area_ref={info['area_km2']} km²")

#  World map showing glacier locations 
fig, ax = plt.subplots(figsize=(16, 8))

try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.plot(ax=ax, color="#E8E8E8", edgecolor="#AAAAAA", linewidth=0.5)
except Exception:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

colors = ["#2196F3", "#E53935", "#4CAF50", "#FF9800", "#9C27B0"]
for i, (name, gdf) in enumerate(glacier_gdfs.items()):
    centroid = gdf.geometry.centroid.values[0]
    ax.plot(centroid.x, centroid.y, "o", color=colors[i],
            markersize=12, zorder=5)
    ax.annotate(
        name.capitalize(),
        xy=(centroid.x, centroid.y),
        xytext=(centroid.x + 3, centroid.y + 3),
        fontsize=10, fontweight="bold", color=colors[i],
        arrowprops=dict(arrowstyle="-", color=colors[i], lw=1.5),
    )

ax.set_title("GlacierCastAI - Study Sites",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "study_sites_map.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: study_sites_map.png")

#  Individual glacier panels 
n    = len(glacier_gdfs)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
if n == 1:
    axes = [axes]

for ax, ((name, gdf), color) in zip(axes, zip(glacier_gdfs.items(), colors)):
    gdf.plot(ax=ax, color=color, alpha=0.4, edgecolor=color.replace("F", "8"),
             linewidth=2)
    info = GLACIERS[name]
    ax.set_title(
        f"{name.capitalize()}\n{info['region']}\n~{info['area_km2']} km²",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")

fig.suptitle("Study Glacier Bounding Boxes (RGI placeholder)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "glacier_bboxes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: glacier_bboxes.png")

print(f"\nDone. Created placeholder polygons for: {list(glacier_gdfs.keys())}")
print("\nNOTE: Replace with RGI 6.0 shapefiles from NSIDC for paper-quality results.")
print("Download: https://nsidc.org/data/nsidc-0770/versions/6")