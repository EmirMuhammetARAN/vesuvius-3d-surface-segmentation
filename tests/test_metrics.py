import numpy as np
import pytest
from src.metrics import (
    compute_surface_dice,
    compute_voi,
    compute_toposcore,
    compute_competition_metric
)

def test_perfect_prediction():
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    gt[10:15, 5:25, 5:25] = 1
    
    result = compute_competition_metric(gt, gt, tau=2.0)
    assert result["surface_dice"] == pytest.approx(1.0, abs=1e-3)
    assert result["voi_score"] == pytest.approx(1.0, abs=1e-3)
    assert result["topo_score"] == pytest.approx(1.0, abs=1e-3)
    assert result["total_score"] == pytest.approx(1.0, abs=1e-3)

def test_hole_penalty():
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    gt[10:15, 5:25, 5:25] = 1
    
    pred = gt.copy()
    # Hollow out a cavity inside
    pred[12, 10:15, 10:15] = 0
    
    result = compute_competition_metric(pred, gt, tau=2.0)
    assert result["total_score"] < 1.0
    assert result["topo_score"] < 1.0

def test_merge_penalty():
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    # Two separate sheets
    gt[5:8, 5:25, 5:25] = 1
    gt[12:15, 5:25, 5:25] = 1
    
    pred = gt.copy()
    # Bridge the two sheets with a connector
    pred[8:12, 10:12, 10:12] = 1
    
    voi_res = compute_voi(pred, gt)
    # Merging two big sheets should produce a merge error
    assert voi_res["voi_merge"] > 0
    assert voi_res["voi_score"] < 1.0
