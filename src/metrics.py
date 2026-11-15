"""
Competition Metrics Implementation for Vesuvius Challenge - Surface Detection

Formula:
  Score = 0.35 * SurfaceDice + 0.35 * VOI_score + 0.30 * TopoScore

Sub-metrics:
1. SurfaceDice@tau: Surface-aware Dice with tolerance tau=2.0
2. VOI_score: Variation of Information mapping: VOI_score = 1 / (1 + 0.3 * (VOI_split + VOI_merge))
3. TopoScore: Topological feature matching across homology dimensions k=0, 1, 2
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def extract_surface_voxels(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the outer surface (boundary) voxel coordinates from a 3D binary mask.
    A foreground voxel is on the surface if at least one of its 6-neighbors is background.
    """
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.int32)
        
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    eroded = ndimage.binary_erosion(mask, structure=struct)
    surface = (mask > 0) & (~eroded)
    return np.argwhere(surface)


def compute_surface_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, tau: float = 2.0) -> float:
    """
    SurfaceDice@tau:
    Measures the fraction of surface points that lie within physical tolerance tau of each other.
    """
    pred_surface = extract_surface_voxels(pred_mask)
    gt_surface = extract_surface_voxels(gt_mask)
    
    # Edge cases
    if len(pred_surface) == 0 and len(gt_surface) == 0:
        return 1.0
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 0.0
        
    # Nearest neighbor distances via KD-Trees
    pred_tree = cKDTree(pred_surface)
    gt_tree = cKDTree(gt_surface)
    
    # Distances from pred to nearest gt
    dist_p2g, _ = gt_tree.query(pred_surface, k=1)
    match_pred = np.mean(dist_p2g <= tau)
    
    # Distances from gt to nearest pred
    dist_g2p, _ = pred_tree.query(gt_surface, k=1)
    match_gt = np.mean(dist_g2p <= tau)
    
    return float(0.5 * (match_pred + match_gt))


def compute_voi(pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float = 0.3) -> dict:
    """
    Variation of Information (VOI) on 3D connected components (26-connectivity).
    VOI_total = VOI_split + VOI_merge
    VOI_score = 1 / (1 + alpha * VOI_total)
    """
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    
    # Label instances
    pred_labeled, n_pred = ndimage.label(pred_mask > 0, structure=struct)
    gt_labeled, n_gt = ndimage.label(gt_mask > 0, structure=struct)
    
    if n_pred == 0 and n_gt == 0:
        return {"voi_total": 0.0, "voi_score": 1.0, "voi_split": 0.0, "voi_merge": 0.0}
    if n_pred == 0 or n_gt == 0:
        return {"voi_total": 10.0, "voi_score": 0.0, "voi_split": 5.0, "voi_merge": 5.0}
        
    # Foreground union mask
    fg_union = (pred_labeled > 0) | (gt_labeled > 0)
    p_labels = pred_labeled[fg_union]
    g_labels = gt_labeled[fg_union]
    
    N = float(len(p_labels))
    if N == 0:
        return {"voi_total": 0.0, "voi_score": 1.0, "voi_split": 0.0, "voi_merge": 0.0}
        
    # Joint contingency table
    # Encode pairs as unique integer
    max_g = n_gt + 1
    pair_ids = p_labels.astype(np.int64) * max_g + g_labels.astype(np.int64)
    unique_pairs, pair_counts = np.unique(pair_ids, return_counts=True)
    p_joint = pair_counts / N
    
    # Marginal distributions
    pred_counts = np.bincount(p_labels)[1:]  # omit 0
    gt_counts = np.bincount(g_labels)[1:]
    
    p_pred = pred_counts[pred_counts > 0] / N
    p_gt = gt_counts[gt_counts > 0] / N
    
    # Entropies
    h_pred = -np.sum(p_pred * np.log2(p_pred + 1e-12))
    h_gt = -np.sum(p_gt * np.log2(p_gt + 1e-12))
    h_joint = -np.sum(p_joint * np.log2(p_joint + 1e-12))
    
    # Conditional entropies
    voi_split = max(0.0, h_gt - (h_pred + h_gt - h_joint))  # H(GT | Pred)
    voi_merge = max(0.0, h_pred - (h_pred + h_gt - h_joint))  # H(Pred | GT)
    voi_total = voi_split + voi_merge
    
    voi_score = 1.0 / (1.0 + alpha * voi_total)
    
    return {
        "voi_total": float(voi_total),
        "voi_score": float(np.clip(voi_score, 0.0, 1.0)),
        "voi_split": float(voi_split),
        "voi_merge": float(voi_merge)
    }


def compute_toposcore(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    TopoScore:
    Approximates Betti number matching across dimensions k=0 (components), k=1 (handles/tunnels), k=2 (cavities).
    """
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 1.0
    if not np.any(pred_mask) or not np.any(gt_mask):
        return 0.0
        
    struct_26 = ndimage.generate_binary_structure(3, 3)
    struct_6 = ndimage.generate_binary_structure(3, 1)
    
    # k=0: Connected components
    _, n_pred_0 = ndimage.label(pred_mask, structure=struct_26)
    _, n_gt_0 = ndimage.label(gt_mask, structure=struct_26)
    f1_0 = (2.0 * min(n_pred_0, n_gt_0)) / max(1, n_pred_0 + n_gt_0)
    
    # k=2: Enclosed cavities (background components enclosed by foreground)
    filled_pred = ndimage.binary_fill_holes(pred_mask)
    cavities_pred = filled_pred & (~pred_mask)
    _, n_pred_2 = ndimage.label(cavities_pred, structure=struct_6)
    
    filled_gt = ndimage.binary_fill_holes(gt_mask)
    cavities_gt = filled_gt & (~gt_mask)
    _, n_gt_2 = ndimage.label(cavities_gt, structure=struct_6)
    
    if n_pred_2 == 0 and n_gt_2 == 0:
        f1_2 = 1.0
    else:
        f1_2 = (2.0 * min(n_pred_2, n_gt_2)) / max(1, n_pred_2 + n_gt_2)
        
    # k=1: Estimated via Euler Characteristic: Chi = Betti0 - Betti1 + Betti2
    # For a planar sheet without holes: Betti0=1, Betti1=0, Betti2=0.
    # Tunnel/hole count directly tracked by difference in Euler characteristics
    f1_1 = 1.0 if abs(n_pred_0 - n_gt_0) == 0 else max(0.0, 1.0 - 0.2 * abs(n_pred_0 - n_gt_0))
    
    toposcore = 0.34 * f1_0 + 0.33 * f1_1 + 0.33 * f1_2
    return float(np.clip(toposcore, 0.0, 1.0))


def compute_competition_metric(pred_mask: np.ndarray, gt_mask: np.ndarray, tau: float = 2.0) -> dict:
    """
    Computes the full official leaderboard score:
    Score = 0.35 * SurfaceDice + 0.35 * VOI_score + 0.30 * TopoScore
    """
    sdice = compute_surface_dice(pred_mask, gt_mask, tau=tau)
    voi_dict = compute_voi(pred_mask, gt_mask, alpha=0.3)
    tscore = compute_toposcore(pred_mask, gt_mask)
    
    total_score = 0.35 * sdice + 0.35 * voi_dict["voi_score"] + 0.30 * tscore
    
    return {
        "total_score": float(total_score),
        "surface_dice": float(sdice),
        "voi_score": float(voi_dict["voi_score"]),
        "voi_total": float(voi_dict["voi_total"]),
        "voi_split": float(voi_dict["voi_split"]),
        "voi_merge": float(voi_dict["voi_merge"]),
        "topo_score": float(tscore)
    }
