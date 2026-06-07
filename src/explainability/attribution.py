"""
Explainability layer for GlacierCastAI.

Two complementary attribution methods:

    1. GradCAM++ - spatial
       Which image regions drive the boundary prediction?
       Applied to the last convolutional layer of the encoder.

    2. SHAP KernelExplainer - tabular
       Which climate and terrain features drive retreat rate?
       Applied to the climate + terrain input vector.

Together these answer the core XAI question:
    "WHY is this glacier retreating faster?"

Output feeds directly into paper Section 5 (Driver Attribution)
and Figure 4 (SHAP plots) and Figure 5 (GradCAM++ maps).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Feature name definitions ───────────────────────────────────────────────
# Order must match climate feature vector in era5.py

CLIMATE_FEATURE_NAMES = [
    "temp_djf",      "temp_mam",      "temp_jja",      "temp_son",
    "precip_djf",    "precip_mam",    "precip_jja",    "precip_son",
    "snowfall_djf",  "snowfall_mam",  "snowfall_jja",  "snowfall_son",
    "radiation_djf", "radiation_mam", "radiation_jja", "radiation_son",
]

TERRAIN_FEATURE_NAMES = [
    "elevation_mean",
    "slope_mean",
    "aspect_sin",
    "aspect_cos",
]

ALL_FEATURE_NAMES = CLIMATE_FEATURE_NAMES + TERRAIN_FEATURE_NAMES


def compute_gradcam_plusplus(
    model: nn.Module,
    input_sequence: torch.Tensor,
    climate_seq: torch.Tensor,
    dem: torch.Tensor,
    target_layer_name: str = "encoder.model.layer4",
) -> np.ndarray:
    """
    GradCAM++ attribution map on the spatial encoder.

    Highlights which spatial regions of the input image most
    strongly drive the glacier boundary prediction.

    Args:
        model: Trained GlacierCastAI model (eval mode).
        input_sequence: (1, T, C, H, W) input tensor.
        climate_seq: (1, T, F) climate tensor.
        dem: (1, 3, H, W) terrain tensor.
        target_layer_name: Dotted path to target conv layer.

    Returns:
        (H, W) attribution heatmap normalized to [0, 1].
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

        # Get target layer
        target_layer = _get_layer(model, target_layer_name)
        if target_layer is None:
            logger.warning(f"Layer {target_layer_name} not found")
            return np.zeros((1, 1))

        # GradCAM++ needs a single image input
        # Use last timestep of the sequence
        img = input_sequence[0, -1].unsqueeze(0)   # (1, C, H, W)

        # Wrap model to accept single image
        class SingleFrameWrapper(nn.Module):
            def __init__(self, full_model, climate, dem, seq):
                super().__init__()
                self.full_model = full_model
                self.climate    = climate
                self.dem        = dem
                self.seq        = seq

            def forward(self, x):
                # Replace last frame with x
                seq = self.seq.clone()
                seq[0, -1] = x[0]
                out = self.full_model(seq, self.climate, self.dem)
                return out["mask"].mean(dim=[2, 3])   # (B, 1)

        wrapper = SingleFrameWrapper(model, climate_seq, dem, input_sequence)
        cam = GradCAMPlusPlus(model=wrapper, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=img)
        return grayscale_cam[0]

    except ImportError:
        logger.warning("grad-cam not installed. Run: pip install grad-cam")
        return np.zeros((1, 1))


def compute_shap_values(
    predict_fn,
    climate_terrain_features: np.ndarray,
    background_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    nsamples: int = 100,
) -> Dict[str, float]:
    """
    SHAP KernelExplainer for climate and terrain tabular inputs.

    Quantifies how much each climate/terrain variable contributes
    to the retreat rate prediction for a single glacier.

    Args:
        predict_fn: Callable (X: np.ndarray) → retreat_rate scalar.
        climate_terrain_features: (1, F) feature vector to explain.
        background_data: (N, F) representative background dataset.
        feature_names: Names for each feature. Uses ALL_FEATURE_NAMES if None.
        nsamples: Number of SHAP samples (higher = more accurate, slower).

    Returns:
        dict mapping feature_name → SHAP value.
        Positive = drives retreat. Negative = buffers retreat.
    """
    try:
        import shap

        if feature_names is None:
            feature_names = ALL_FEATURE_NAMES

        explainer   = shap.KernelExplainer(predict_fn, background_data)
        shap_values = explainer.shap_values(
            climate_terrain_features,
            nsamples=nsamples,
            silent=True,
        )

        return {
            name: float(val)
            for name, val in zip(feature_names, shap_values[0])
        }

    except ImportError:
        logger.warning("shap not installed. Run: pip install shap")
        return {}


def generate_driver_report(
    shap_values: Dict[str, float],
    glacier_name: str,
    horizon_years: int = 5,
    top_k: int = 3,
) -> str:
    """
    Generate human-readable retreat driver attribution report.

    This is the output that bridges ML and glaciological communication.
    Forms the basis of paper Section 5 and the XAI figures.

    Example output:
        Glacier : Aletsch | Horizon: 5yr
        Top 3 retreat drivers (SHAP attribution):
          1. temp_jja     : +0.841  (38.2%)  [↑ accelerates retreat]
          2. aspect_sin   : +0.312  (14.2%)  [↑ accelerates retreat]
          3. snowfall_djf : -0.228  (10.4%)  [↓ buffers retreat]

    Args:
        shap_values: Feature → SHAP value dict from compute_shap_values.
        glacier_name: Human-readable glacier name.
        horizon_years: Prediction horizon in years.
        top_k: Number of top drivers to report.

    Returns:
        Formatted report string.
    """
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_features = sorted_features[:top_k]
    total_abs    = sum(abs(v) for _, v in shap_values.items()) + 1e-8

    lines = [
        f"Glacier : {glacier_name} | Horizon: {horizon_years}yr",
        f"Top {top_k} retreat drivers (SHAP attribution):",
    ]

    for rank, (feat, val) in enumerate(top_features, 1):
        direction = "↑ accelerates retreat" if val > 0 else "↓ buffers retreat"
        pct       = 100 * abs(val) / total_abs
        lines.append(
            f"  {rank}. {feat:<20}: {val:+.3f}  ({pct:.1f}%)  [{direction}]"
        )

    return "\n".join(lines)


def explain_glacier(
    model: nn.Module,
    sample: Dict,
    background_data: np.ndarray,
    glacier_name: str,
    horizon_years: int = 5,
    device: str = "cuda",
) -> Dict:
    """
    Full explanation pipeline for a single glacier sample.

    Runs both GradCAM++ and SHAP and returns all outputs
    ready for paper figure generation.

    Args:
        model: Trained GlacierCastAI model.
        sample: Single batch dict from GlacierSequenceDataset.
        background_data: (N, F) SHAP background dataset.
        glacier_name: Glacier identifier.
        horizon_years: Prediction horizon.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with:
            'gradcam_map'   : (H, W) spatial attribution heatmap
            'shap_values'   : dict of feature → SHAP value
            'driver_report' : formatted string report
    """
    model.eval()
    model = model.to(device)

    image_seq   = sample["image_seq"].unsqueeze(0).to(device)
    climate_seq = sample["climate_seq"].unsqueeze(0).to(device)
    dem         = sample["dem"].unsqueeze(0).to(device)

    # GradCAM++ spatial attribution
    gradcam_map = compute_gradcam_plusplus(
        model=model,
        input_sequence=image_seq,
        climate_seq=climate_seq,
        dem=dem,
    )

    # SHAP climate/terrain attribution
    climate_np = sample["climate_seq"].mean(axis=0).numpy()  # (F,) mean over T
    terrain_np = np.array([
        sample["dem"][0].mean().item(),   # elevation_mean
        sample["dem"][1].mean().item(),   # slope_mean
        sample["dem"][2].mean().item(),   # aspect_sin
        sample["dem"][2].std().item(),    # aspect_cos (proxy)
    ])
    features = np.concatenate([climate_np, terrain_np])[None]  # (1, F)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            climate_t = torch.from_numpy(
                X[:, :len(CLIMATE_FEATURE_NAMES)]
            ).float().unsqueeze(1).expand(-1, image_seq.shape[1], -1).to(device)
            out = model(image_seq.expand(X.shape[0], -1, -1, -1, -1), climate_t, dem.expand(X.shape[0], -1, -1, -1))
            return out["retreat"].mean(dim=1).cpu().numpy()

    shap_values = compute_shap_values(
        predict_fn=predict_fn,
        climate_terrain_features=features,
        background_data=background_data,
    )

    report = generate_driver_report(shap_values, glacier_name, horizon_years)

    logger.info(f"\n{report}")

    return {
        "gradcam_map":   gradcam_map,
        "shap_values":   shap_values,
        "driver_report": report,
    }


def _get_layer(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """Get a submodule by dotted path string."""
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module