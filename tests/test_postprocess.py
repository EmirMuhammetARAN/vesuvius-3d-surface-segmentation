import numpy as np
import pytest
from src.postprocess import (
    remove_small_components,
    plug_small_holes_6c,
    binary_closing_sphere,
    fill_cavities,
    full_postprocess_pipeline
)

def test_remove_small_components():
    vol = np.zeros((30, 30, 30), dtype=np.uint8)
    # Big component (e.g. 5x5x5 = 125 voxels)
    vol[5:10, 5:10, 5:10] = 1
    # Tiny noise component (1 voxel)
    vol[25, 25, 25] = 1
    
    cleaned = remove_small_components(vol, min_voxels=50)
    assert cleaned[25, 25, 25] == 0, "Tiny noise was not removed"
    assert np.sum(cleaned[5:10, 5:10, 5:10]) == 125, "Big component was altered"

def test_plug_small_holes():
    # 3x3x3 cube with the center voxel empty (0)
    vol = np.ones((5, 5, 5), dtype=np.uint8)
    vol[2, 2, 2] = 0  # 1-voxel hole trapped inside
    
    plugged = plug_small_holes_6c(vol)
    assert plugged[2, 2, 2] == 1, "1-voxel hole was not plugged"

def test_fill_cavities():
    # Hollow sphere / shell (cavity inside)
    vol = np.zeros((15, 15, 15), dtype=np.uint8)
    vol[3:12, 3:12, 3:12] = 1
    vol[5:10, 5:10, 5:10] = 0  # hollow cavity
    
    filled = fill_cavities(vol)
    assert np.all(filled[5:10, 5:10, 5:10] == 1), "Cavity was not filled"

def test_full_pipeline():
    # Test end-to-end pipeline execution on a mock 3D sheet
    prob_map = np.zeros((40, 40, 40), dtype=np.float32)
    # Sheet from z=18 to z=22
    prob_map[18:22, 10:30, 10:30] = 0.85
    # Puncture with holes
    prob_map[20, 15, 15] = 0.0
    prob_map[20, 25, 25] = 0.0
    # Add dust specks
    prob_map[5, 5, 5] = 0.9
    
    result = full_postprocess_pipeline(prob_map, threshold=0.35, min_component_voxels=50)
    assert result.shape == prob_map.shape
    assert result.dtype == np.uint8
    assert result[5, 5, 5] == 0, "Dust speck should be filtered"
    assert result[20, 15, 15] == 1, "Punctured hole should be closed/plugged"
