"""
Evaluation metrics for GlacierCastAI.

Standard metrics:
    IoU     - Intersection over Union of predicted vs actual glacier mask
    BF1     - Boundary F1: precision/recall on glacier edge pixels only

Custom metrics (paper contribution):
    RR-RMSE - Retreat Rate RMSE: accuracy of predicted annual area loss

All metrics computed on CPU numpy arrays - called from validation_step
in trainer.py and from evaluate.py for final paper results.
"""

import numpy as np
from scipy.ndimage import binary_erosion
from typing import Dict, List


def compute_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Intersection-over-Union for binary glacier mask.

    Args:
        pred_mask: (H, W) predicted probability map [0, 1].
        target_mask: (H, W) binary ground truth.
        threshold: Binarization threshold.

    Returns:
        IoU score in [0, 1].
    """
    pred_bin   = (pred_mask >= threshold).astype(bool)
    target_bin = target_mask.astype(bool)

    intersection = np.logical_and(pred_bin, target_bin).sum()
    union        = np.logical_or(pred_bin, target_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def compute_boundary_f1(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.5,
    boundary_width: int = 3,
) -> Dict[str, float]:
    """
    Boundary F1 (BF1): precision/recall on glacier edge pixels only.

    Motivation: interior glacier pixels are easy to predict correctly.
    Edge prediction quality determines scientific accuracy of area estimates
    and is what matters for downstream water resource / sea level models.

    Args:
        pred_mask: (H, W) predicted probability map [0, 1].
        target_mask: (H, W) binary ground truth.
        threshold: Binarization threshold.
        boundary_width: Erosion kernel size to extract boundary band.

    Returns:
        dict with 'precision', 'recall', 'f1'.
    """
    pred_bin   = (pred_mask >= threshold).astype(bool)
    target_bin = target_mask.astype(bool)

    struct = np.ones((boundary_width, boundary_width), dtype=bool)

    def get_boundary(mask: np.ndarray) -> np.ndarray:
        eroded = binary_erosion(mask, structure=struct)
        return np.logical_and(mask, ~eroded)

    pred_boundary   = get_boundary(pred_bin)
    target_boundary = get_boundary(target_bin)

    tp = np.logical_and(pred_boundary, target_boundary).sum()
    fp = np.logical_and(pred_boundary, ~target_boundary).sum()
    fn = np.logical_and(~pred_boundary, target_boundary).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def compute_retreat_rate_rmse(
    pred_areas_km2: np.ndarray,
    target_areas_km2: np.ndarray,
) -> float:
    """
    Retreat Rate RMSE (RR-RMSE) - custom metric defined in this paper.

    Measures accuracy of predicted annual glacier area loss in km²/yr.
    This is the core scientific quantity - more meaningful than pixel IoU
    for glaciologists and policymakers.

    Args:
        pred_areas_km2: (N,) predicted glacier areas in km² over time.
        target_areas_km2: (N,) ground truth areas in km² over time.

    Returns:
        RMSE of annual retreat rates in km²/yr.
    """
    pred_rates   = np.diff(pred_areas_km2)
    target_rates = np.diff(target_areas_km2)
    return float(np.sqrt(np.mean((pred_rates - target_rates) ** 2)))


def compute_all_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    pred_retreat: np.ndarray,
    target_retreat: np.ndarray,
    pixel_area_km2: float = 0.0009,   # 30m pixel → 0.0009 km²
) -> Dict[str, float]:
    """
    Compute full evaluation suite for one prediction.

    Called per sample in evaluate.py for final paper results.

    Args:
        pred_mask: (H, W) predicted probability map.
        target_mask: (H, W) binary ground truth.
        pred_retreat: (3,) predicted retreat rates [1yr, 3yr, 5yr].
        target_retreat: (3,) ground truth retreat rates.
        pixel_area_km2: Area of one pixel in km².

    Returns:
        dict with all metrics ready for W&B logging and paper tables.
    """
    iou      = compute_iou(pred_mask, target_mask)
    boundary = compute_boundary_f1(pred_mask, target_mask)

    retreat_mae  = float(np.mean(np.abs(pred_retreat - target_retreat)))
    retreat_rmse = float(np.sqrt(np.mean((pred_retreat - target_retreat) ** 2)))

    # Glacier area from mask (km²)
    pred_area   = pred_mask.mean() * pred_mask.size * pixel_area_km2
    target_area = target_mask.mean() * target_mask.size * pixel_area_km2
    area_error  = abs(pred_area - target_area)

    return {
        "iou":              iou,
        "boundary_f1":      boundary["f1"],
        "boundary_prec":    boundary["precision"],
        "boundary_recall":  boundary["recall"],
        "retreat_mae":      retreat_mae,
        "retreat_rmse":     retreat_rmse,
        "area_error_km2":   area_error,
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate per-sample metrics into dataset-level means and stds.

    Used in evaluate.py to produce the final numbers for paper Table 2.

    Args:
        metrics_list: List of per-sample metric dicts.

    Returns:
        dict with mean and std for each metric.
    """
    aggregated = {}
    keys = metrics_list[0].keys()

    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"]  = float(np.std(values))

    return aggregated