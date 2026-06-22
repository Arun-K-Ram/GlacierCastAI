"""
Preprocessing pipeline for GlacierCastAI.

For each glacier:
1. Load Landsat scenes (green + swir1 bands)
2. Co-register to DEM grid using GDAL
3. Clip to glacier bounding box
4. Compute NDSI + glacier mask
5. Extract 256x256 patches
6. Save as .npz files with image + mask + dem + metadata

Output structure:
    data/processed/patches/<glacier>/<scene_id>_r<row>_c<col>.npz
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import json

LANDSAT_DIR  = Path("data/raw/landsat")
DEM_DIR      = Path("data/raw/dem")
RGI_DIR      = Path("data/raw/rgi")
PATCHES_DIR  = Path("data/processed/patches")
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE   = 256
OVERLAP      = 64
MIN_GLACIER  = 0.03   # min glacier fraction per patch
MIN_VALID    = 0.70   # min non-NaN fraction per patch
PIXEL_AREA   = 0.0009 # km² per 30m pixel

GLACIERS = {
    "aletsch":   {"bbox": [7.8, 46.3, 8.2, 46.7],  "epsg": 32632},
    "gangotri":  {"bbox": [79.0, 30.8, 79.4, 31.1], "epsg": 32644},
    "grey":      {"bbox": [-73.3, -51.0, -72.8, -50.6], "epsg": 32718},
    "columbia":  {"bbox": [-147.2, 61.0, -146.5, 61.4], "epsg": 32606},
    "athabasca": {"bbox": [-117.4, 52.1, -117.1, 52.3], "epsg": 32611},
}


def load_and_normalize_band(path: Path) -> np.ndarray:
    """Load a single band GeoTIFF and normalize to [0, 1]."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    # Landsat Collection 2 Level-2 SR scale: multiply by 0.0000275, add -0.2
    data = data * 0.0000275 - 0.2
    data = np.clip(data, 0, 1)
    return data


def compute_ndsi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    denom = green + swir1
    denom = np.where(denom == 0, 1e-8, denom)
    return (green - swir1) / denom


def load_dem_for_glacier(name: str, target_shape: tuple,
                          target_transform, target_crs) -> np.ndarray:
    """Load and reproject DEM to match Landsat grid."""
    dem_dir  = DEM_DIR / name
    dem_files = list(dem_dir.glob("dem_merged.tif")) or list(dem_dir.glob("*.tif"))

    if not dem_files:
        print(f"  No DEM found for {name}")
        return np.zeros(target_shape, dtype=np.float32)

    dem_path = dem_files[0]
    H, W     = target_shape

    with rasterio.open(dem_path) as src:
        dem_repr = np.zeros((H, W), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_repr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    return dem_repr


def compute_terrain_features(dem: np.ndarray) -> np.ndarray:
    """Return (3, H, W) terrain stack: slope, aspect_sin, aspect_cos."""
    dy, dx      = np.gradient(dem, 30.0)
    slope       = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect      = np.arctan2(-dx, dy)
    aspect_sin  = np.sin(aspect).astype(np.float32)
    aspect_cos  = np.cos(aspect).astype(np.float32)
    slope       = slope.astype(np.float32)
    return np.stack([slope, aspect_sin, aspect_cos], axis=0)  # (3, H, W)

def load_climate_features(glacier_name: str, year: int) -> np.ndarray:
    """
    Load ERA5 climate features for a given glacier and year.
    Returns (16,) feature vector: 4 variables x 4 seasons.
    """
    import xarray as xr

    climate_dir = Path("data/raw/climate") / glacier_name
    file_ua     = climate_dir / "data_stream-moda_stepType-avgua.nc"
    file_ad     = climate_dir / "data_stream-moda_stepType-avgad.nc"

    if not file_ua.exists() or not file_ad.exists():
        return np.zeros(16, dtype=np.float32)

    try:
        ds_ua = xr.open_dataset(file_ua, engine="netcdf4")
        ds_ad = xr.open_dataset(file_ad, engine="netcdf4")

        t2m = ds_ua["t2m"].mean(dim=["latitude", "longitude"]) - 273.15
        tp  = ds_ad["tp"].mean(dim=["latitude", "longitude"]) * 1000
        sf  = ds_ad["sf"].mean(dim=["latitude", "longitude"]) * 1000
        ssr = ds_ad["ssr"].mean(dim=["latitude", "longitude"])

        features = []
        for var in [t2m, tp, sf, ssr]:
            year_data = var.sel(valid_time=var.valid_time.dt.year == year)
            if len(year_data) == 0:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue
            for months in [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]:
                seasonal = year_data.sel(
                    valid_time=year_data.valid_time.dt.month.isin(months)
                )
                val = float(seasonal.mean()) if len(seasonal) > 0 else 0.0
                features.append(val)

        ds_ua.close()
        ds_ad.close()
        return np.array(features[:16], dtype=np.float32)

    except Exception as e:
        print(f"    Climate load failed for {glacier_name} {year}: {e}")
        return np.zeros(16, dtype=np.float32)
    
def extract_patches(image, mask, dem_features, scene_id,
                    glacier_name, year):
    """
    Sliding window patch extraction.
    Returns list of dicts ready to save as .npz.
    """
    C, H, W = image.shape
    stride  = PATCH_SIZE - OVERLAP
    patches = []

    for row in range(0, H - PATCH_SIZE + 1, stride):
        for col in range(0, W - PATCH_SIZE + 1, stride):
            img_p  = image[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            mask_p = mask[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            dem_p  = dem_features[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]

            # Filter: too many NaN pixels
            valid_frac = np.isfinite(img_p).mean()
            if valid_frac < MIN_VALID:
                continue

            # Filter: not enough glacier
            glacier_frac = mask_p.mean()
            if glacier_frac < MIN_GLACIER:
                continue

            patches.append({
                "image":    img_p,
                "mask":     mask_p.astype(np.float32),
                "dem":      dem_p,
                "scene_id": scene_id,
                "glacier":  glacier_name,
                "year":     year,
                "row":      row,
                "col":      col,
            })

    return patches


def process_glacier(name: str, config: dict):
    print(f"\n{'='*50}")
    print(f"PREPROCESSING: {name.upper()}")
    print(f"{'='*50}")

    landsat_dir = LANDSAT_DIR / name
    out_dir     = PATCHES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = [d for d in landsat_dir.iterdir() if d.is_dir()]
    scene_dirs.sort()
    print(f"  Found {len(scene_dirs)} scenes")

    all_patches  = 0
    scene_meta   = []

    for scene_dir in scene_dirs:
        green_path = scene_dir / "green.tif"
        swir1_path = scene_dir / "swir1.tif"

        if not green_path.exists() or not swir1_path.exists():
            continue

        scene_id = scene_dir.name
        # Extract year from scene ID e.g. LC08_L2SP_195027_20150830_...
        try:
            year = int(scene_id.split("_")[3][:4])
        except Exception:
            year = 0

        print(f"  Processing {scene_id} ({year})...")

        # Load bands
        green = load_and_normalize_band(green_path)
        swir1 = load_and_normalize_band(swir1_path)

        # Get spatial reference from green band
        with rasterio.open(green_path) as src:
            profile   = src.profile.copy()
            transform = src.transform
            crs       = src.crs
            H, W      = src.height, src.width

        # Compute NDSI and glacier mask
        ndsi         = compute_ndsi(green, swir1)
        glacier_mask = (ndsi > 0.4).astype(np.float32)

        # Stack image channels: green, swir1, ndsi
        image = np.stack([green, swir1, ndsi], axis=0)  # (3, H, W)

        # Load DEM co-registered to this scene
        dem      = load_dem_for_glacier(name, (H, W), transform, crs)
        dem_feat = compute_terrain_features(dem)   # (3, H, W)

        # Extract patches
        patches = extract_patches(
            image, glacier_mask, dem_feat,
            scene_id, name, year,
        )

        # Load climate ONCE per scene not per patch
        climate = load_climate_features(name, year)

        # Save patches
        for p in patches:
            fname = out_dir / f"{scene_id}_r{p['row']}_c{p['col']}.npz"
            np.savez_compressed(
                fname,
                image=p["image"],
                mask=p["mask"],
                dem=p["dem"],
                year=np.array([p["year"]]),
                climate=climate,
            )

        all_patches += len(patches)
        scene_meta.append({
            "scene_id":     scene_id,
            "year":         year,
            "glacier_area": float(glacier_mask.sum() * PIXEL_AREA),
            "patches":      len(patches),
        })

        print(f"    Patches: {len(patches)} | "
              f"Area: {glacier_mask.sum() * PIXEL_AREA:.1f} km²")

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"glacier": name, "scenes": scene_meta}, f, indent=2)

    print(f"\n  Total patches: {all_patches}")
    print(f"  Metadata saved: {meta_path}")
    return scene_meta


#  Run all glaciers 
all_meta = {}
for name, config in GLACIERS.items():
    meta = process_glacier(name, config)
    all_meta[name] = meta

#  Summary 
print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)
total_patches = 0
for name, meta in all_meta.items():
    n_scenes  = len(meta)
    n_patches = sum(m["patches"] for m in meta)
    total_patches += n_patches
    print(f"  {name:<12}: {n_scenes:>3} scenes | {n_patches:>5} patches")
print(f"  {'TOTAL':<12}: {'':>3}        | {total_patches:>5} patches")