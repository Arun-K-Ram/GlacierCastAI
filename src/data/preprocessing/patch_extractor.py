"""
Patch extraction from co-registered glacier scenes.

Extracts 256x256 patches using a sliding window approach.
Patches with insufficient glacier coverage are discarded.
Output is saved as numpy arrays for fast loading during training.

Patch naming convention:
    <glacier>_<scene_id>_r<row>_c<col>.npz
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    overlap: int = 64,
    min_glacier_fraction: float = 0.05,
    min_valid_fraction: float = 0.7,
) -> list[dict]:
    """
    Extract patches from a single scene using sliding window.

    Args:
        image: (C, H, W) float32 image array - may contain NaN (clouds).
        mask: (H, W) binary glacier mask (1=glacier, 0=background).
        patch_size: Spatial size of each patch in pixels.
        overlap: Overlap between adjacent patches in pixels.
        min_glacier_fraction: Minimum fraction of patch that must be
                              glacier to keep the patch.
        min_valid_fraction: Minimum fraction of non-NaN pixels required.
                            Patches with too many clouds are discarded.

    Returns:
        List of dicts, each with keys:
            'image': (C, patch_size, patch_size) float32
            'mask':  (patch_size, patch_size) uint8
            'row':   int - top-left row index
            'col':   int - top-left col index
    """
    _, H, W = image.shape
    stride = patch_size - overlap
    patches = []

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            img_patch = image[:, row:row + patch_size, col:col + patch_size]
            mask_patch = mask[row:row + patch_size, col:col + patch_size]

            # Skip patches with too many clouds/NaN
            valid_frac = np.isfinite(img_patch).mean()
            if valid_frac < min_valid_fraction:
                continue

            # Skip patches with insufficient glacier coverage
            glacier_frac = mask_patch.mean()
            if glacier_frac < min_glacier_fraction:
                continue

            patches.append({
                "image": img_patch,
                "mask": mask_patch.astype(np.uint8),
                "row": row,
                "col": col,
            })

    return patches


def save_patches(
    patches: list[dict],
    output_dir: Path,
    glacier_name: str,
    scene_id: str,
    dem_patch: Optional[np.ndarray] = None,
    climate_features: Optional[np.ndarray] = None,
) -> list[Path]:
    """
    Save extracted patches as compressed numpy archives.

    Each .npz file contains:
        - image: (C, H, W) spectral bands
        - mask: (H, W) binary glacier mask
        - dem: (3, H, W) terrain features (slope, aspect_sin, aspect_cos)
        - climate: (F,) climate feature vector for this timestep

    Args:
        patches: Output of extract_patches.
        output_dir: Directory to save .npz files.
        glacier_name: Glacier identifier.
        scene_id: Scene identifier (used in filename).
        dem_patch: (3, H, W) terrain features, same spatial extent as scene.
        climate_features: (F,) climate vector for this scene's date.

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for patch in patches:
        row, col = patch["row"], patch["col"]
        filename = output_dir / f"{glacier_name}_{scene_id}_r{row}_c{col}.npz"

        save_dict = {
            "image": patch["image"],
            "mask": patch["mask"],
        }

        if dem_patch is not None:
            dem_crop = dem_patch[
                :,
                row:row + patch["image"].shape[1],
                col:col + patch["image"].shape[2],
            ]
            save_dict["dem"] = dem_crop

        if climate_features is not None:
            save_dict["climate"] = climate_features

        np.savez_compressed(filename, **save_dict)
        saved.append(filename)

    return saved


def build_sequence_index(
    patches_dir: Path,
    glacier_name: str,
    seq_len: int = 8,
    horizon: int = 1,
) -> list[dict]:
    """
    Build a list of training sequences from saved patches.

    Each sequence is a list of T consecutive patches from the same
    spatial location, paired with a target patch H years in the future.

    Args:
        patches_dir: Directory containing .npz patch files.
        glacier_name: Glacier to build sequences for.
        seq_len: Number of input timesteps T.
        horizon: Prediction horizon in years.

    Returns:
        List of dicts, each with:
            'input_paths': list of T .npz file paths
            'target_path': .npz file path for the target timestep
            'glacier': glacier name
            'horizon': prediction horizon in years
    """
    # Group patches by spatial location (row, col)
    all_files = sorted(patches_dir.glob(f"{glacier_name}_*.npz"))

    location_groups: dict[str, list] = {}

    for f in all_files:
        # Parse: <glacier>_<scene_id>_r<row>_c<col>.npz
        parts = f.stem.split("_")
        row_col = f"{parts[-2]}_{parts[-1]}"  # e.g. "r128_c64"

        if row_col not in location_groups:
            location_groups[row_col] = []
        location_groups[row_col].append(f)

    sequences = []

    for loc, files in location_groups.items():
        files = sorted(files)  # sort by scene_id (which encodes date)

        # Slide window over temporal axis
        for i in range(len(files) - seq_len - horizon + 1):
            input_paths = files[i: i + seq_len]
            target_path = files[i + seq_len + horizon - 1]

            sequences.append({
                "input_paths": input_paths,
                "target_path": target_path,
                "glacier": glacier_name,
                "horizon": horizon,
            })

    logger.info(
        f"{glacier_name}: {len(sequences)} sequences "
        f"(T={seq_len}, horizon={horizon}yr)"
    )
    return sequences