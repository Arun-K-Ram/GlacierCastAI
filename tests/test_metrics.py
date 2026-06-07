"""
Unit tests for evaluation metrics.

Run with: poetry run pytest tests/ -v
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_iou,
    compute_boundary_f1,
    compute_retreat_rate_rmse,
    compute_all_metrics,
    aggregate_metrics,
)


#  Fixtures ─

@pytest.fixture
def perfect_mask():
    """Square glacier mask — same pred and target."""
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[32:96, 32:96] = 1.0
    return mask


@pytest.fixture
def shifted_mask():
    """Slightly shifted glacier mask — partial overlap."""
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[40:100, 40:100] = 1.0
    return mask


@pytest.fixture
def empty_mask():
    return np.zeros((128, 128), dtype=np.float32)


@pytest.fixture
def full_mask():
    return np.ones((128, 128), dtype=np.float32)


#  IoU 

def test_iou_perfect(perfect_mask):
    assert compute_iou(perfect_mask, perfect_mask) == pytest.approx(1.0)


def test_iou_no_overlap(perfect_mask, empty_mask):
    # pred has glacier, target has none
    assert compute_iou(perfect_mask, empty_mask) == pytest.approx(0.0)


def test_iou_both_empty(empty_mask):
    # Both empty → perfect agreement
    assert compute_iou(empty_mask, empty_mask) == pytest.approx(1.0)


def test_iou_partial(perfect_mask, shifted_mask):
    iou = compute_iou(perfect_mask, shifted_mask)
    assert 0.0 < iou < 1.0


def test_iou_range(perfect_mask, shifted_mask):
    iou = compute_iou(perfect_mask, shifted_mask)
    assert 0.0 <= iou <= 1.0


def test_iou_threshold(perfect_mask):
    # Probabilities just below threshold should give IoU = 0
    low_prob = perfect_mask * 0.4
    assert compute_iou(low_prob, perfect_mask, threshold=0.5) == pytest.approx(0.0)


#  Boundary F1 

def test_boundary_f1_perfect(perfect_mask):
    result = compute_boundary_f1(perfect_mask, perfect_mask)
    assert result["f1"]        == pytest.approx(1.0, abs=0.01)
    assert result["precision"] == pytest.approx(1.0, abs=0.01)
    assert result["recall"]    == pytest.approx(1.0, abs=0.01)


def test_boundary_f1_range(perfect_mask, shifted_mask):
    result = compute_boundary_f1(perfect_mask, shifted_mask)
    assert 0.0 <= result["f1"]        <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"]    <= 1.0


def test_boundary_f1_keys(perfect_mask):
    result = compute_boundary_f1(perfect_mask, perfect_mask)
    assert "precision" in result
    assert "recall"    in result
    assert "f1"        in result


def test_boundary_f1_no_overlap(perfect_mask, empty_mask):
    result = compute_boundary_f1(perfect_mask, empty_mask)
    assert result["f1"] == pytest.approx(0.0, abs=0.01)


#  Retreat Rate RMSE 

def test_rr_rmse_perfect():
    areas = np.array([100.0, 95.0, 90.0, 85.0])
    assert compute_retreat_rate_rmse(areas, areas) == pytest.approx(0.0)


def test_rr_rmse_positive():
    pred   = np.array([100.0, 94.0, 89.0, 83.0])
    target = np.array([100.0, 95.0, 90.0, 85.0])
    rmse = compute_retreat_rate_rmse(pred, target)
    assert rmse > 0.0


def test_rr_rmse_units():
    # 1 km²/yr consistent error
    pred   = np.array([100.0, 94.0, 88.0])
    target = np.array([100.0, 95.0, 90.0])
    rmse = compute_retreat_rate_rmse(pred, target)
    assert rmse == pytest.approx(1.0, abs=0.01)


#  compute_all_metrics 

def test_compute_all_metrics_keys(perfect_mask):
    result = compute_all_metrics(
        pred_mask=perfect_mask,
        target_mask=perfect_mask,
        pred_retreat=np.array([5.0, 10.0, 15.0]),
        target_retreat=np.array([5.0, 10.0, 15.0]),
    )
    expected_keys = [
        "iou", "boundary_f1", "boundary_prec",
        "boundary_recall", "retreat_mae",
        "retreat_rmse", "area_error_km2",
    ]
    for key in expected_keys:
        assert key in result


def test_compute_all_metrics_perfect(perfect_mask):
    result = compute_all_metrics(
        pred_mask=perfect_mask,
        target_mask=perfect_mask,
        pred_retreat=np.array([5.0, 10.0, 15.0]),
        target_retreat=np.array([5.0, 10.0, 15.0]),
    )
    assert result["iou"]          == pytest.approx(1.0)
    assert result["retreat_mae"]  == pytest.approx(0.0)
    assert result["retreat_rmse"] == pytest.approx(0.0)
    assert result["area_error_km2"] == pytest.approx(0.0, abs=1e-6)


#  aggregate_metrics 

def test_aggregate_metrics_keys():
    metrics_list = [
        {"iou": 0.8, "boundary_f1": 0.75},
        {"iou": 0.9, "boundary_f1": 0.85},
    ]
    aggregated = aggregate_metrics(metrics_list)
    assert "iou_mean"          in aggregated
    assert "iou_std"           in aggregated
    assert "boundary_f1_mean"  in aggregated
    assert "boundary_f1_std"   in aggregated


def test_aggregate_metrics_values():
    metrics_list = [
        {"iou": 0.8},
        {"iou": 0.9},
    ]
    aggregated = aggregate_metrics(metrics_list)
    assert aggregated["iou_mean"] == pytest.approx(0.85)
    assert aggregated["iou_std"]  == pytest.approx(0.05, abs=0.001)