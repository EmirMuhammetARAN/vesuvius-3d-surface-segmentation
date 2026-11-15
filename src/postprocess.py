"""
5-Step Topological Post-Processing Pipeline (1st Place Vesuvius Solution)

Ablation impact from 1st place solution:
- Raw Model: 0.572 (Public) / 0.596 (Private)
- + Step 0: Remove small components (<20k voxels) -> 0.586 / 0.614
- + Step 1: Plug 1-voxel holes (Watertight neighborhood) -> 0.598 / 0.622
- + Step 2: Patch large holes (Heightmap interpolation) -> 0.601 / 0.625
- + Step 3: Binary closing (Radius-3 sphere) -> 0.606 / 0.627
- + Step 4: Binary fill holes -> 0.606 / 0.627 (Rank 1)
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from skimage import morphology


def create_spherical_footprint(radius: int = 3) -> np.ndarray:
    """Creates a spherical 3D structuring element of given radius."""
    L = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(L, L, L, indexing="ij")
    return (X**2 + Y**2 + Z**2 <= radius**2).astype(bool)


def remove_small_components(mask: np.ndarray, min_voxels: int = 20000) -> np.ndarray:
    """
    Step 0: Removes disconnected foreground noise components with voxel count < min_voxels.
    Uses 26-connectivity (generate_binary_structure with rank=3, connectivity=3).
    """
    struct = ndimage.generate_binary_structure(3, 3)
    labeled, num_features = ndimage.label(mask, structure=struct)
    if num_features == 0:
        return mask
    
    counts = np.bincount(labeled.ravel())
    # Identify labels exceeding threshold (label 0 is background)
    valid_labels = np.where(counts >= min_voxels)[0]
    valid_labels = valid_labels[valid_labels != 0]
    
    clean_mask = np.isin(labeled, valid_labels)
    return clean_mask.astype(np.uint8)


def plug_small_holes_6c(mask: np.ndarray) -> np.ndarray:
    """
    Step 1: Plugs 1-voxel and tiny holes by checking local 3D neighborhood.
    Any 0-voxel that is trapped (e.g. >= 5 of its 6-neighbors are 1, or >= 22 of 26-neighbors)
    is plugged to ensure 6-connected watertightness without bloating the sheet.
    """
    # 6-connectivity kernel
    kernel_6 = ndimage.generate_binary_structure(3, 1).astype(int)
    kernel_6[1, 1, 1] = 0  # omit center
    
    neighbor_count_6 = ndimage.convolve(mask.astype(int), kernel_6, mode="constant", cval=0)
    
    # 26-connectivity kernel
    kernel_26 = np.ones((3, 3, 3), dtype=int)
    kernel_26[1, 1, 1] = 0
    neighbor_count_26 = ndimage.convolve(mask.astype(int), kernel_26, mode="constant", cval=0)
    
    output = mask.copy()
    # A hole voxel has mask == 0 but is heavily surrounded by foreground
    hole_condition = (mask == 0) & ((neighbor_count_6 >= 5) | (neighbor_count_26 >= 22))
    output[hole_condition] = 1
    return output


def binary_closing_sphere(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Step 2: Applies morphological binary closing with a spherical footprint.
    Closes gaps and cracks of radius <= radius without distorting flat geometry.
    """
    footprint = create_spherical_footprint(radius)
    closed = ndimage.binary_closing(mask, structure=footprint)
    return closed.astype(np.uint8)


def patch_large_holes_heightmap(sheet_mask: np.ndarray) -> np.ndarray:
    """
    Step 3: Represents a single sheet as a 2D heightmap along its dominant projection plane
    and fills internal gaps using linear interpolation across gap edges along X and Y axes.
    """
    coords = np.argwhere(sheet_mask > 0)
    if len(coords) < 100:
        return sheet_mask
    
    # Determine dominant projection axis (axis with smallest variance / thickness)
    stds = np.std(coords, axis=0)
    proj_axis = int(np.argmin(stds))  # axis perpendicular to sheet
    u_axis, v_axis = [i for i in range(3) if i != proj_axis]
    
    # Create 2D bounding box
    u_min, u_max = coords[:, u_axis].min(), coords[:, u_axis].max()
    v_min, v_max = coords[:, v_axis].min(), coords[:, v_axis].max()
    
    # Extract median height for each (u, v) point
    u_idx = coords[:, u_axis] - u_min
    v_idx = coords[:, v_axis] - v_min
    h_val = coords[:, proj_axis]
    
    grid_shape = (u_max - u_min + 1, v_max - v_min + 1)
    if grid_shape[0] * grid_shape[1] > 2_000_000:
        # Avoid excessive memory for massive bounding boxes
        return sheet_mask
        
    height_grid = np.full(grid_shape, np.nan, dtype=np.float32)
    # Assign known points
    height_grid[u_idx, v_idx] = h_val
    
    # Interpolate gaps inside convex hull
    known_mask = ~np.isnan(height_grid)
    # Find internal holes in 2D projection
    holes_2d = ndimage.binary_fill_holes(known_mask) & (~known_mask)
    if not np.any(holes_2d):
        return sheet_mask  # No holes to patch
    
    points = np.argwhere(known_mask)
    values = height_grid[known_mask]
    hole_points = np.argwhere(holes_2d)
    
    try:
        interp_vals = griddata(points, values, hole_points, method="linear")
        # Replace remaining NaNs with nearest
        nan_mask = np.isnan(interp_vals)
        if np.any(nan_mask):
            interp_vals[nan_mask] = griddata(points, values, hole_points[nan_mask], method="nearest")
            
        patched_mask = sheet_mask.copy()
        for (hu, hv), hz in zip(hole_points, interp_vals):
            orig_u = hu + u_min
            orig_v = hv + v_min
            orig_z = int(np.round(hz))
            
            target_coord = [0, 0, 0]
            target_coord[proj_axis] = orig_z
            target_coord[u_axis] = orig_u
            target_coord[v_axis] = orig_v
            
            # Bounds check
            if (0 <= target_coord[0] < sheet_mask.shape[0] and
                0 <= target_coord[1] < sheet_mask.shape[1] and
                0 <= target_coord[2] < sheet_mask.shape[2]):
                patched_mask[tuple(target_coord)] = 1
                
        return patched_mask
    except Exception:
        return sheet_mask


def fill_cavities(mask: np.ndarray) -> np.ndarray:
    """
    Step 4: Fills arbitrary-size enclosed 3D cavities (k=2 holes) via binary_fill_holes.
    """
    filled = ndimage.binary_fill_holes(mask)
    return filled.astype(np.uint8)


def full_postprocess_pipeline(
    raw_pred: np.ndarray,
    threshold: float = 0.35,
    min_component_voxels: int = 20000,
    closing_radius: int = 3,
    use_heightmap: bool = True
) -> np.ndarray:
    """
    Executes the complete 1st place post-processing pipeline.
    
    Parameters:
    -----------
    raw_pred: 3D probability map [0.0, 1.0] or binary mask
    threshold: probability cutoff (1st place found 0.35-0.40 optimal for private LB)
    min_component_voxels: threshold for pruning dust specks
    closing_radius: spherical closing radius
    use_heightmap: whether to run 2D heightmap gap repair on individual sheets
    
    Returns:
    --------
    cleaned_mask: uint8 3D binary mask (0 or 1)
    """
    # 1. Binarize
    if raw_pred.dtype != bool and raw_pred.dtype != np.uint8:
        binary_mask = (raw_pred >= threshold).astype(np.uint8)
    else:
        binary_mask = (raw_pred > 0).astype(np.uint8)
        
    # Step 0: Purge small noise components (< 20K voxels)
    mask = remove_small_components(binary_mask, min_voxels=min_component_voxels)
    if not np.any(mask):
        return mask
        
    # Step 1: Spherical binary closing (radius 3)
    if closing_radius > 0:
        mask = binary_closing_sphere(mask, radius=closing_radius)
        
    # Step 2: Per-sheet heightmap interpolation (repair large tears)
    if use_heightmap:
        struct = ndimage.generate_binary_structure(3, 3)
        labeled, num_features = ndimage.label(mask, structure=struct)
        repaired = np.zeros_like(mask)
        
        for sheet_id in range(1, num_features + 1):
            sheet_mask = (labeled == sheet_id).astype(np.uint8)
            sheet_patched = patch_large_holes_heightmap(sheet_mask)
            repaired |= sheet_patched
        mask = repaired
        
    # Step 3: Plug remaining 1-voxel holes (Watertight neighborhood)
    mask = plug_small_holes_6c(mask)
    
    # Step 4: Fill enclosed 3D cavities (k=2)
    mask = fill_cavities(mask)
    
    return mask.astype(np.uint8)
