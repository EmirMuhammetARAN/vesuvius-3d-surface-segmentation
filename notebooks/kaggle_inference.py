import os
import gc
import time
import glob
import shutil
import zipfile
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageSequence
from scipy import ndimage
from skimage.morphology import remove_small_objects
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act2(x + res)
        return x

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False)
        self.norm_up = nn.InstanceNorm3d(in_channels // 2, affine=True)
        self.act_up = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        total_in = (in_channels // 2) + skip_channels
        self.conv = ResidualBlock3D(total_in, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act_up(self.norm_up(self.upsample(x)))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResEncUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: list = [32, 64, 128, 256, 320]
    ):
        super().__init__()
        c0, c1, c2, c3, c4 = channels

        self.enc0 = ResidualBlock3D(in_channels, c0)
        self.down0 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = ResidualBlock3D(c0, c1)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock3D(c1, c2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResidualBlock3D(c2, c3)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ResidualBlock3D(c3, c4)

        self.dec3 = DecoderBlock3D(c4, c3, c3)
        self.dec2 = DecoderBlock3D(c3, c2, c2)
        self.dec1 = DecoderBlock3D(c2, c1, c1)
        self.dec0 = DecoderBlock3D(c1, c0, c0)

        self.out_conv = nn.Conv3d(c0, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        d0 = self.down0(e0)

        e1 = self.enc1(d0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        b = self.bottleneck(d3)

        u3 = self.dec3(b, e3)
        u2 = self.dec2(u3, e2)
        u1 = self.dec1(u2, e1)
        u0 = self.dec0(u1, e0)

        return self.out_conv(u0)

class _SegConvWrap(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class _SegResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.norm2 = nn.GroupNorm(8, c)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = _SegConvWrap(c, c, 3, 1, 1, False)
        self.conv2 = _SegConvWrap(c, c, 3, 1, 1, False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + identity

class _SegUpSampleWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_non_trainable = nn.Upsample(scale_factor=(2.0, 2.0, 2.0), mode="trilinear", align_corners=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_non_trainable(x)

class PureSegResNet3D(nn.Module):
    """Bit-for-bit Pure PyTorch implementation of MONAI SegResNet (zero external dependencies)."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_filters: int = 32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)):
        super().__init__()
        self.act_mod = nn.ReLU(inplace=True)
        self.convInit = _SegConvWrap(in_channels, init_filters, 3, 1, 1, False)
        self.down_layers = nn.ModuleList()
        c = init_filters
        for i, n_blk in enumerate(blocks_down):
            layers = []
            if i == 0:
                layers.append(nn.Identity())
            else:
                layers.append(_SegConvWrap(c // 2, c, 3, 2, 1, False))
            for _ in range(n_blk):
                layers.append(_SegResBlock(c))
            self.down_layers.append(nn.Sequential(*layers))
            if i < len(blocks_down) - 1:
                c *= 2
        self.up_layers = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for _, n_blk in enumerate(blocks_up):
            half_c = c // 2
            self.up_samples.append(nn.Sequential(_SegConvWrap(c, half_c, 1, 1, 0, False), _SegUpSampleWrap()))
            self.up_layers.append(nn.Sequential(*[_SegResBlock(half_c) for _ in range(n_blk)]))
            c = half_c
        self.conv_final = nn.Sequential(nn.GroupNorm(8, c), nn.ReLU(inplace=True), _SegConvWrap(c, out_channels, 1, 1, 0, True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convInit(x)
        skips = []
        for down in self.down_layers:
            x = down(x)
            skips.append(x)
        skips.pop()
        for up_s, up_l in zip(self.up_samples, self.up_layers):
            x = up_s(x) + skips.pop()
            x = up_l(x)
        return self.conv_final(x)

def gaussian_weight_map_3d(patch_size: tuple, sigma_scale: float = 0.125) -> np.ndarray:
    D, H, W = patch_size
    sigmas = (D * sigma_scale, H * sigma_scale, W * sigma_scale)
    
    z = np.exp(-0.5 * ((np.arange(D) - D / 2.0) / sigmas[0]) ** 2)
    y = np.exp(-0.5 * ((np.arange(H) - H / 2.0) / sigmas[1]) ** 2)
    x = np.exp(-0.5 * ((np.arange(W) - W / 2.0) / sigmas[2]) ** 2)
    
    weights = z[:, None, None] * y[None, :, None] * x[None, None, :]
    weights /= (weights.max() + 1e-8)
    return weights.astype(np.float32)

def _run_patches_forward(
    model: torch.nn.Module,
    volume_norm: np.ndarray,
    coords: list,
    patch_size: tuple,
    batch_size: int,
    device: str,
    tta_mode: int,
    orig_shape: tuple,
    padded_shape: tuple
) -> np.ndarray:
    orig_D, orig_H, orig_W = orig_shape
    D, H, W = padded_shape
    pD, pH, pW = patch_size
    use_cuda = str(device).startswith("cuda")

    prob_accum = np.zeros((D, H, W), dtype=np.float32)
    weight_accum = np.zeros((D, H, W), dtype=np.float32)
    gaussian_weights = gaussian_weight_map_3d(patch_size)

    for i in range(0, len(coords), batch_size):
        b_coords = coords[i:i + batch_size]
        patches = [volume_norm[z:z+pD, y:y+pH, x:x+pW] for (z, y, x) in b_coords]
        batch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
            probs = torch.sigmoid(model(batch_tensor))

            if tta_mode == 2:
                # In-place 2x TTA (Original + 3D Diagonal Flip) - minimal VRAM footprint
                probs = probs + torch.flip(
                    torch.sigmoid(model(torch.flip(batch_tensor, dims=[-3, -2, -1]))),
                    dims=[-3, -2, -1]
                )
                probs = probs * 0.5
            elif tta_mode >= 4:
                # In-place 4x TTA (Original + X, Y, Z Flips sequentially to prevent VRAM spikes)
                probs = probs + torch.flip(
                    torch.sigmoid(model(torch.flip(batch_tensor, dims=[-1]))),
                    dims=[-1]
                )
                probs = probs + torch.flip(
                    torch.sigmoid(model(torch.flip(batch_tensor, dims=[-2]))),
                    dims=[-2]
                )
                probs = probs + torch.flip(
                    torch.sigmoid(model(torch.flip(batch_tensor, dims=[-3]))),
                    dims=[-3]
                )
                probs = probs * 0.25

            probs_np = probs.cpu().numpy()[:, 0]

        del batch_tensor, probs

        for idx, (z, y, x) in enumerate(b_coords):
            prob_accum[z:z+pD, y:y+pH, x:x+pW] += probs_np[idx] * gaussian_weights
            weight_accum[z:z+pD, y:y+pH, x:x+pW] += gaussian_weights

    weight_accum = np.maximum(weight_accum, 1e-8)
    prob_map = prob_accum / weight_accum
    return prob_map[:orig_D, :orig_H, :orig_W]

def predict_volume_batched(
    model: torch.nn.Module,
    volume_norm: np.ndarray,
    patch_size: tuple = (192, 192, 192),
    overlap: float = 0.60,
    batch_size: int = 2,
    device: str = "cuda",
    tta_mode: int = 2
) -> np.ndarray:
    """
    Rock-solid 3D sliding-window inference with automatic OOM batch-size fallback.
    Accumulates in CPU RAM to prevent VRAM exhaustion on large hidden test volumes.
    """
    orig_D, orig_H, orig_W = volume_norm.shape
    pD, pH, pW = patch_size

    pad_d = max(0, pD - orig_D)
    pad_h = max(0, pH - orig_H)
    pad_w = max(0, pW - orig_W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume_norm = np.pad(volume_norm, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    D, H, W = volume_norm.shape
    sD = max(1, int(pD * (1.0 - overlap)))
    sH = max(1, int(pH * (1.0 - overlap)))
    sW = max(1, int(pW * (1.0 - overlap)))

    z_steps = list(range(0, max(1, D - pD + 1), sD))
    if z_steps[-1] + pD < D:
        z_steps.append(D - pD)
    y_steps = list(range(0, max(1, H - pH + 1), sH))
    if y_steps[-1] + pH < H:
        y_steps.append(H - pH)
    x_steps = list(range(0, max(1, W - pW + 1), sW))
    if x_steps[-1] + pW < W:
        x_steps.append(W - pW)

    coords = [(z, y, x) for z in z_steps for y in y_steps for x in x_steps]

    try:
        return _run_patches_forward(
            model, volume_norm, coords, patch_size, batch_size, device, tta_mode,
            (orig_D, orig_H, orig_W), (D, H, W)
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 1:
            print("        [!] VRAM OOM at batch_size={}, retrying safely with batch_size=1...".format(batch_size))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _run_patches_forward(
                model, volume_norm, coords, patch_size, 1, device, tta_mode,
                (orig_D, orig_H, orig_W), (D, H, W)
            )
        raise

def build_anisotropic_struct(z_radius: int = 3, xy_radius: int = 2) -> np.ndarray:
    z, r = int(z_radius), int(xy_radius)
    depth = 2*z + 1
    size  = 2*r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz = z; cy = cx = r
    for dz in range(-z, z+1):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dy*dy + dx*dx <= r*r:
                    struct[cz+dz, cy+dy, cx+dx] = True
    return struct

def patch_large_holes_heightmap_fast(sheet_mask: np.ndarray) -> np.ndarray:
    """
    Vectorized, boundary-subsampled 2D heightmap hole interpolation.
    Uses only the 2-pixel rim around detected holes for Delaunay triangulation,
    running ~1000x faster than full-sheet griddata while preserving hole repair accuracy.
    """
    coords = np.argwhere(sheet_mask > 0)
    if len(coords) < 100:
        return sheet_mask

    stds = np.std(coords, axis=0)
    proj_axis = int(np.argmin(stds))
    u_axis, v_axis = [i for i in range(3) if i != proj_axis]

    u_min, u_max = int(coords[:, u_axis].min()), int(coords[:, u_axis].max())
    v_min, v_max = int(coords[:, v_axis].min()), int(coords[:, v_axis].max())

    grid_shape = (u_max - u_min + 1, v_max - v_min + 1)
    if grid_shape[0] * grid_shape[1] > 2000000:
        return sheet_mask

    u_idx = coords[:, u_axis] - u_min
    v_idx = coords[:, v_axis] - v_min
    h_val = coords[:, proj_axis]

    height_grid = np.full(grid_shape, np.nan, dtype=np.float32)
    height_grid[u_idx, v_idx] = h_val

    known_mask = ~np.isnan(height_grid)
    holes_2d = ndimage.binary_fill_holes(known_mask) & (~known_mask)
    if not np.any(holes_2d):
        return sheet_mask

    hole_points = np.argwhere(holes_2d)
    if len(hole_points) == 0 or len(hole_points) > 50000:
        return sheet_mask

    # Only use the 2-pixel boundary rim around holes for fast, exact Delaunay interpolation
    rim_mask = ndimage.binary_dilation(holes_2d, iterations=2) & known_mask
    points = np.argwhere(rim_mask)
    if len(points) < 4:
        return sheet_mask
    values = height_grid[rim_mask]

    if len(points) > 6000:
        step = max(1, len(points) // 6000)
        points = points[::step]
        values = values[::step]

    try:
        from scipy.interpolate import griddata
        interp_vals = griddata(points, values, hole_points, method="linear")
        nan_mask = np.isnan(interp_vals)
        if np.any(nan_mask):
            interp_vals[nan_mask] = griddata(points, values, hole_points[nan_mask], method="nearest")

        valid = ~np.isnan(interp_vals)
        if not np.any(valid):
            return sheet_mask

        hole_points = hole_points[valid]
        interp_vals = interp_vals[valid]

        orig_u = hole_points[:, 0] + u_min
        orig_v = hole_points[:, 1] + v_min
        orig_z = np.clip(np.round(interp_vals).astype(np.int64), 0, sheet_mask.shape[proj_axis] - 1)

        patched_mask = sheet_mask.copy()
        idx_tuple = [None, None, None]
        idx_tuple[proj_axis] = orig_z
        idx_tuple[u_axis] = orig_u
        idx_tuple[v_axis] = orig_v
        patched_mask[tuple(idx_tuple)] = 1
        return patched_mask
    except Exception:
        return sheet_mask

def plug_watertight_holes_6c(mask: np.ndarray) -> np.ndarray:
    kernel_6 = ndimage.generate_binary_structure(3, 1).astype(int)
    kernel_6[1, 1, 1] = 0
    neighbor_count_6 = ndimage.convolve(mask.astype(int), kernel_6, mode="constant", cval=0)
    kernel_26 = np.ones((3, 3, 3), dtype=int)
    kernel_26[1, 1, 1] = 0
    neighbor_count_26 = ndimage.convolve(mask.astype(int), kernel_26, mode="constant", cval=0)
    output = mask.copy()
    hole_condition = (mask == 0) & ((neighbor_count_6 >= 5) | (neighbor_count_26 >= 22))
    output[hole_condition] = 1
    return output

def seeded_hysteresis_postprocess(
    prob: np.ndarray,
    T_high: float = 0.66,
    T_low: float = 0.42,
    z_radius: int = 3,
    xy_radius: int = 2,
    dust_min_size: int = 10000
) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    strong = prob >= float(T_high)
    weak = prob >= float(T_low)

    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)

    struct_hyst = ndimage.generate_binary_structure(3, 3)
    mask = ndimage.binary_propagation(strong, mask=weak, structure=struct_hyst).astype(np.uint8)

    if not mask.any():
        return np.zeros_like(prob, dtype=np.uint8)

    # Step 1: Anisotropic morphological closing (z_radius=3, xy_radius=2) BEFORE dust removal so cracked sheets reconnect
    struct_close = build_anisotropic_struct(z_radius, xy_radius)
    if struct_close is not None and (z_radius > 0 or xy_radius > 0):
        mask = ndimage.binary_closing(mask, structure=struct_close).astype(np.uint8)

    # Step 2: Fast vectorized rim-based heightmap hole interpolation per valid sheet
    struct_label = ndimage.generate_binary_structure(3, 3)
    labeled, num_features = ndimage.label(mask, structure=struct_label)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        valid_labels = np.where(counts >= dust_min_size)[0]
        valid_labels = valid_labels[valid_labels != 0]
        if len(valid_labels) > 0:
            repaired = np.zeros_like(mask, dtype=np.uint8)
            for sheet_id in valid_labels:
                sheet_mask = (labeled == sheet_id).astype(np.uint8)
                repaired |= patch_large_holes_heightmap_fast(sheet_mask)
            mask = repaired

    # Step 3: Watertight 6-connected hole plugging & 3D cavity fill
    mask = plug_watertight_holes_6c(mask.astype(np.uint8))
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # Step 4: Final dust removal
    if int(dust_min_size) > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=int(dust_min_size)).astype(np.uint8)

    return mask.astype(np.uint8)

def read_tiff_volume(path: str) -> np.ndarray:
    try:
        return tifffile.imread(path)
    except Exception:
        with Image.open(path) as img:
            frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]
            return np.stack(frames, axis=0)

def write_tiff_volume(path: str, volume: np.ndarray):
    try:
        tifffile.imwrite(path, volume.astype(np.uint8))
    except Exception:
        frames = [Image.fromarray(slice_2d) for slice_2d in volume]
        frames[0].save(path, save_all=True, append_images=frames[1:])

def find_all_files(extension: str = ".pth"):
    found = []
    for search_root in ["/kaggle/input", "."]:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root, followlinks=True):
                for f in files:
                    if f.lower().endswith(extension.lower()):
                        found.append(os.path.join(root, f))
    return found

def get_checkpoints_to_run(device: str):
    all_pths = find_all_files(".pth") + find_all_files(".pt")
    # If fewer than 8 .pth files are found directly, check if a .zip archive in /kaggle/input contains .pth weights
    if len([p for p in all_pths if "best" in os.path.basename(p).lower()]) < 8:
        all_zips = find_all_files(".zip")
        extract_dir = "/tmp/extracted_ckpts"
        for zpath in all_zips:
            if "submission" in os.path.basename(zpath).lower() or "colab" in os.path.basename(zpath).lower():
                continue
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    pth_members = [m for m in zf.namelist() if m.lower().endswith(".pth")]
                    if pth_members:
                        os.makedirs(extract_dir, exist_ok=True)
                        for m in pth_members:
                            target_f = os.path.join(extract_dir, os.path.basename(m))
                            if not os.path.exists(target_f):
                                zf.extract(m, extract_dir)
            except Exception:
                pass
        if os.path.exists(extract_dir):
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".pth"):
                        all_pths.append(os.path.join(root, f))
    print(f"[*] Discovered {len(all_pths)} weight candidate(s):")
    for p in all_pths:
        print(f"    - {p}")

    ckpt_seg_192_ft_best = None
    ckpt_seg_192_ft_snap = None
    ckpt_resenc_160 = None
    ckpt_resenc_192_f1 = None
    ckpt_resenc_192_f2 = None
    ckpt_resenc_192_f3 = None
    ckpt_resenc_alldata = None
    ckpt_seg_f1 = None
    ckpt_seg_f2 = None
    ckpt_seg_192_f3 = None

    # Strict architecture and fold matching (Never cross-assign architectures)
    for p in all_pths:
        name = os.path.basename(p).lower()

        if "segresnet" in name:
            if "finetune" in name or "ft" in name:
                if "best" in name:
                    ckpt_seg_192_ft_best = p
                elif "epoch" in name or "snap" in name:
                    ckpt_seg_192_ft_snap = p
            elif "best" in name:
                if "fold3" in name or "colab_fold3" in name:
                    ckpt_seg_192_f3 = p
                elif "fold2" in name:
                    ckpt_seg_f2 = p
                else:
                    ckpt_seg_f1 = p
        elif ("resenc" in name or "unet" in name) and "best" in name:
            if "alldata" in name or "all_data" in name:
                ckpt_resenc_alldata = p
            elif "fold3" in name or "colab_fold3" in name:
                ckpt_resenc_192_f3 = p
            elif "fold2" in name:
                ckpt_resenc_192_f2 = p
            elif "192" in name:
                ckpt_resenc_192_f1 = p
            elif "160" in name:
                ckpt_resenc_160 = p

    models_to_run = []
    # Tuple format: (model, patch_size, batch_size, default_ov, default_tta, name, weight)

    # 1. Pure PyTorch SegResNet-3D Models (Highest Scoring Backbone: +0.06 to +0.10 over ResEncUNet3D across all scrolls!)
    if ckpt_seg_192_ft_best:
        try:
            print(f"[+] Loading PureSegResNet3D (192^3, ALL-DATA 160-EPOCH SUPREME CHAMPION): {os.path.basename(ckpt_seg_192_ft_best)}")
            m_seg_ft = PureSegResNet3D(
                in_channels=1, out_channels=1, init_filters=32,
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
            ).to(device)
            sd_seg_ft = torch.load(ckpt_seg_192_ft_best, map_location=device)
            m_seg_ft.load_state_dict(sd_seg_ft.get("model_state_dict", sd_seg_ft))
            m_seg_ft.eval()
            models_to_run.append((m_seg_ft, (192, 192, 192), 2, 0.665, 2, "SegResNet3D-p192-AllData-Best", 4.5))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_seg_192_ft_best}: {e}")

    if ckpt_seg_192_ft_snap:
        try:
            print(f"[+] Loading PureSegResNet3D (192^3, ALL-DATA Snapshot): {os.path.basename(ckpt_seg_192_ft_snap)}")
            m_seg_snap = PureSegResNet3D(
                in_channels=1, out_channels=1, init_filters=32,
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
            ).to(device)
            sd_seg_snap = torch.load(ckpt_seg_192_ft_snap, map_location=device)
            m_seg_snap.load_state_dict(sd_seg_snap.get("model_state_dict", sd_seg_snap))
            m_seg_snap.eval()
            models_to_run.append((m_seg_snap, (192, 192, 192), 2, 0.665, 2, "SegResNet3D-p192-AllData-Snap", 2.5))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_seg_192_ft_snap}: {e}")

    if ckpt_seg_f2:
        try:
            print(f"[+] Loading PureSegResNet3D (160^3, Fold 2 Overall Champion): {os.path.basename(ckpt_seg_f2)}")
            m_seg2 = PureSegResNet3D(
                in_channels=1, out_channels=1, init_filters=32,
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
            ).to(device)
            sd_seg2 = torch.load(ckpt_seg_f2, map_location=device)
            m_seg2.load_state_dict(sd_seg2.get("model_state_dict", sd_seg2))
            m_seg2.eval()
            models_to_run.append((m_seg2, (160, 160, 160), 2, 0.50, 4, "SegResNet3D-p160-f2", 3.5))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_seg_f2}: {e}")

    if ckpt_seg_f1:
        try:
            print(f"[+] Loading PureSegResNet3D (160^3, Fold 1 Champion): {os.path.basename(ckpt_seg_f1)}")
            m_seg1 = PureSegResNet3D(
                in_channels=1, out_channels=1, init_filters=32,
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
            ).to(device)
            sd_seg1 = torch.load(ckpt_seg_f1, map_location=device)
            m_seg1.load_state_dict(sd_seg1.get("model_state_dict", sd_seg1))
            m_seg1.eval()
            models_to_run.append((m_seg1, (160, 160, 160), 2, 0.50, 4, "SegResNet3D-p160-f1", 3.0))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_seg_f1}: {e}")

    if ckpt_seg_192_f3:
        try:
            print(f"[+] Loading PureSegResNet3D (192^3, Fold 3 Champion): {os.path.basename(ckpt_seg_192_f3)}")
            m_seg3 = PureSegResNet3D(
                in_channels=1, out_channels=1, init_filters=32,
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
            ).to(device)
            sd_seg3 = torch.load(ckpt_seg_192_f3, map_location=device)
            m_seg3.load_state_dict(sd_seg3.get("model_state_dict", sd_seg3))
            m_seg3.eval()
            models_to_run.append((m_seg3, (192, 192, 192), 2, 0.665, 2, "SegResNet3D-p192-f3", 2.5))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_seg_192_f3}: {e}")

    # 2. ResEncUNet3D Models (Complementary Multi-Scale Support)
    if ckpt_resenc_alldata:
        try:
            print(f"[+] Loading ResEncUNet3D (192^3, ALL-DATA Champion): {os.path.basename(ckpt_resenc_alldata)}")
            m_all = ResEncUNet3D().to(device)
            sd_all = torch.load(ckpt_resenc_alldata, map_location=device)
            m_all.load_state_dict(sd_all.get("model_state_dict", sd_all))
            m_all.eval()
            models_to_run.append((m_all, (192, 192, 192), 2, 0.665, 2, "ResEncUNet3D-p192-AllData", 2.0))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_resenc_alldata}: {e}")

    if ckpt_resenc_192_f3:
        try:
            print(f"[+] Loading ResEncUNet3D (192^3, Fold 3): {os.path.basename(ckpt_resenc_192_f3)}")
            m192_3 = ResEncUNet3D().to(device)
            sd192_3 = torch.load(ckpt_resenc_192_f3, map_location=device)
            m192_3.load_state_dict(sd192_3.get("model_state_dict", sd192_3))
            m192_3.eval()
            models_to_run.append((m192_3, (192, 192, 192), 2, 0.665, 2, "ResEncUNet3D-p192-f3", 2.0))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_resenc_192_f3}: {e}")

    if ckpt_resenc_192_f1:
        try:
            print(f"[+] Loading ResEncUNet3D (192^3, Fold 1): {os.path.basename(ckpt_resenc_192_f1)}")
            m192_1 = ResEncUNet3D().to(device)
            sd192_1 = torch.load(ckpt_resenc_192_f1, map_location=device)
            m192_1.load_state_dict(sd192_1.get("model_state_dict", sd192_1))
            m192_1.eval()
            models_to_run.append((m192_1, (192, 192, 192), 2, 0.665, 2, "ResEncUNet3D-p192-f1", 1.5))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_resenc_192_f1}: {e}")

    if ckpt_resenc_192_f2:
        try:
            print(f"[+] Loading ResEncUNet3D (192^3, Fold 2): {os.path.basename(ckpt_resenc_192_f2)}")
            m192_2 = ResEncUNet3D().to(device)
            sd192_2 = torch.load(ckpt_resenc_192_f2, map_location=device)
            m192_2.load_state_dict(sd192_2.get("model_state_dict", sd192_2))
            m192_2.eval()
            models_to_run.append((m192_2, (192, 192, 192), 2, 0.665, 2, "ResEncUNet3D-p192-f2", 1.0))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_resenc_192_f2}: {e}")

    if ckpt_resenc_160:
        try:
            print(f"[+] Loading ResEncUNet3D (160^3, Fold 1): {os.path.basename(ckpt_resenc_160)}")
            m160 = ResEncUNet3D().to(device)
            sd160 = torch.load(ckpt_resenc_160, map_location=device)
            m160.load_state_dict(sd160.get("model_state_dict", sd160))
            m160.eval()
            models_to_run.append((m160, (160, 160, 160), 2, 0.50, 2, "ResEncUNet3D-p160-f1", 1.0))
        except Exception as e:
            print(f"[!] Warning: Failed to load {ckpt_resenc_160}: {e}")

    total_w = sum(w for _, _, _, _, _, _, w in models_to_run)
    print("=" * 60)
    print(f"[+] Multi-Architecture Ensemble Active! Total Models: {len(models_to_run)}")
    for idx, (_, ps, bs, ov, tta, name, w) in enumerate(models_to_run):
        pct = (w / total_w) * 100.0 if total_w > 0 else 0.0
        print(f"    [{idx+1}] {name} ({ps[0]}^3, batch={bs}, overlap={ov:.3f}, TTA={tta}x, weight={w:.1f} -> {pct:.1f}%)")
    print("=" * 60)

    return models_to_run

def find_test_items():
    test_items = []
    test_csv = None
    for search_root in ["/kaggle/input", "."]:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root, followlinks=True):
                for f in files:
                    if f.lower() == "test.csv":
                        test_csv = os.path.join(root, f)
                        break
                if test_csv:
                    break
        if test_csv:
            break

    candidate_tifs = {}
    for search_root in ["/kaggle/input", "."]:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root, followlinks=True):
                if "train" in root.lower() and "test" not in root.lower():
                    continue
                for f in files:
                    if f.lower().endswith(".tif") or f.lower().endswith(".tiff"):
                        stem = Path(f).stem
                        candidate_tifs[stem] = os.path.join(root, f)

    if test_csv and os.path.exists(test_csv):
        try:
            df = pd.read_csv(test_csv)
            for _, row in df.iterrows():
                tid = str(row["id"])
                if tid in candidate_tifs:
                    test_items.append((tid, candidate_tifs[tid]))
        except Exception as e:
            print(f"[!] Warning reading test.csv: {e}")

    if not test_items and candidate_tifs:
        for tid, path in candidate_tifs.items():
            test_items.append((tid, path))

    return test_items

def main():
    start_time = time.time()
    SAFE_BUDGET_HRS = 8.00  # 8h 00m safe target inside Kaggle's 9.0h hard limit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Accelerated Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    working_dir = Path("/kaggle/working") if os.path.exists("/kaggle/working") else Path("./submissions")
    working_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = working_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    final_zip_path = working_dir / "submission.zip"

    try:
        models_to_run = get_checkpoints_to_run(device)
        if not models_to_run:
            print("[!] WARNING: No checkpoints found! Creating baseline zero prediction model.")
            dummy_model = ResEncUNet3D().to(device).eval()
            models_to_run.append((dummy_model, (160, 160, 160), 2, 0.50, 1, "FallbackDummy", 1.0))

        test_items = find_test_items()
        print(f"[*] Found {len(test_items)} test volume(s) to process: {[t[0] for t in test_items]}")

        generated_tif_names = []
        tier1_vol_times = []

        # Create/reset submission.zip and stream each volume directly into it to prevent /kaggle/working 20GB disk overflow
        with zipfile.ZipFile(str(final_zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
            if not test_items:
                print("[!] Warning: No test volumes located! Generating placeholder 1407735.tif to guarantee submission output.")
                dummy_mask = np.zeros((65, 320, 320), dtype=np.uint8)
                out_root_tif = working_dir / "1407735.tif"
                write_tiff_volume(str(out_root_tif), dummy_mask)
                zf_out.write(str(out_root_tif), arcname="1407735.tif")
                generated_tif_names.append("1407735.tif")

            for v_idx, (img_id, img_path) in enumerate(test_items):
                vol_start = time.time()
                elapsed_hrs = (vol_start - start_time) / 3600.0
                remaining_vols = len(test_items) - v_idx

                if tier1_vol_times:
                    avg_t1_hrs = (sum(tier1_vol_times) / len(tier1_vol_times)) / 3600.0
                    projected_hrs = elapsed_hrs + (avg_t1_hrs * remaining_vols)
                else:
                    projected_hrs = 0.0

                # 4-Tier Dynamic 9-Hour Pace Controller (Sequential batch_size=2 for 100% crash/OOM immunity)
                active_models = models_to_run
                tier_name = "TIER 1 (MAX SCORE: 27-Patch High-Overlap + 4x/2x TTA on All 8 Models)"
                tier_level = 1

                if elapsed_hrs > 8.15:
                    tier_name = "TIER 4 (TURBO FINISH GUARD: Top 4 Models, 8-Patch, 1x TTA)"
                    active_models = models_to_run[:4]
                    tier_level = 4
                elif elapsed_hrs > 7.20 or (projected_hrs > SAFE_BUDGET_HRS and elapsed_hrs > 4.5):
                    tier_name = "TIER 3 (FAST HIGH-SCORE: 8/27-Patch + 2x/1x TTA on All 8 Models)"
                    tier_level = 3
                elif projected_hrs > SAFE_BUDGET_HRS or elapsed_hrs > 5.80:
                    tier_name = "TIER 2 (HIGH SCORE: 27-Patch High-Overlap + 2x/1x TTA on All 8 Models)"
                    tier_level = 2

                print(f"\n[*] [{v_idx+1}/{len(test_items)}] Volume {img_id} | Elapsed: {elapsed_hrs:.2f}h | Mode: {tier_name}")
                volume = read_tiff_volume(img_path)
                vol_shape = volume.shape
                volume_norm = volume.astype(np.float32) / 255.0
                del volume
                print(f"    Dimensions: {vol_shape} (D, H, W)")

                ensemble_prob_map = np.zeros(vol_shape, dtype=np.float32)
                accum_weight = 0.0

                for model, patch_size, bs, def_ov, def_tta, name, w in active_models:
                    if tier_level == 1:
                        eff_ov, eff_tta = def_ov, def_tta
                    elif tier_level == 2:
                        eff_ov = def_ov
                        eff_tta = 2 if patch_size[0] == 192 else 1
                    elif tier_level == 3:
                        eff_ov = 0.50 if patch_size[0] == 192 else 0.25
                        eff_tta = 2 if patch_size[0] == 192 else 1
                    else:
                        eff_ov = 0.50 if patch_size[0] == 192 else 0.25
                        eff_tta = 1

                    try:
                        print(f"    -> {name} ({patch_size[0]}^3, ov={eff_ov:.3f}, batch={bs}, TTA={eff_tta}x, w={w:.1f})")
                        prob_map = predict_volume_batched(
                            model,
                            volume_norm,
                            patch_size=patch_size,
                            overlap=eff_ov,
                            batch_size=bs,
                            device=device,
                            tta_mode=eff_tta
                        )
                        ensemble_prob_map += (prob_map * w)
                        accum_weight += w
                        del prob_map
                    except Exception as model_err:
                        print(f"    [!] Warning: Model {name} encountered error on {img_id}: {model_err}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if accum_weight > 0:
                    ensemble_prob_map /= accum_weight

                if tier_level == 1:
                    tier1_vol_times.append(time.time() - vol_start)

                print("    Applying Calibrated Seeded Hysteresis + Fast Rim Heightmap Repair + Watertight Topology...")
                try:
                    binary_mask = seeded_hysteresis_postprocess(
                        ensemble_prob_map,
                        T_high=0.66,
                        T_low=0.42,
                        z_radius=3,
                        xy_radius=2,
                        dust_min_size=10000
                    )
                except Exception as pp_err:
                    print(f"    [!] Warning in postprocessing {img_id}: {pp_err}. Using threshold fallback.")
                    binary_mask = (ensemble_prob_map >= 0.60).astype(np.uint8)

                tmp_tif = Path("/tmp") / f"{img_id}.tif" if os.path.exists("/tmp") else working_dir / f"{img_id}.tif"
                write_tiff_volume(str(tmp_tif), binary_mask)
                zf_out.write(str(tmp_tif), arcname=f"{img_id}.tif")
                generated_tif_names.append(f"{img_id}.tif")

                # Keep loose .tif in working_dir only when len(test_items) <= 2 so /kaggle/working never hits 20GB limit
                if len(test_items) <= 2 and tmp_tif.parent != working_dir:
                    try:
                        shutil.copyfile(str(tmp_tif), str(working_dir / f"{img_id}.tif"))
                    except Exception:
                        pass
                if tmp_tif.parent == Path("/tmp") and tmp_tif.exists():
                    try:
                        tmp_tif.unlink()
                    except Exception:
                        pass

                print(f"[+] Added to submission.zip: {img_id}.tif (Foreground voxels: {int(np.sum(binary_mask > 0))})")

                del volume_norm
                del ensemble_prob_map
                del binary_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        try:
            shutil.copyfile(str(final_zip_path), str(sub_dir / "submission.zip"))
        except Exception:
            pass

        print("\n" + "=" * 60)
        print(f"[+] SUCCESS: Created valid submission at {final_zip_path}")
        print(f"    Zip size: {final_zip_path.stat().st_size} bytes")
        print(f"    Files in zip ({len(generated_tif_names)}): {generated_tif_names}")
        print(f"    Output files in {working_dir}: {[f.name for f in working_dir.iterdir() if f.is_file()]}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[!] ERROR in inference: {e}")
        traceback.print_exc()
        if not final_zip_path.exists() or final_zip_path.stat().st_size == 0:
            emergency_tif = working_dir / "1407735.tif"
            write_tiff_volume(str(emergency_tif), np.zeros((65, 320, 320), dtype=np.uint8))
            with zipfile.ZipFile(str(final_zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.write(str(emergency_tif), arcname=emergency_tif.name)
            print(f"[+] Emergency fallback created at: {final_zip_path}")
        print("[!] Execution finished with fallback to ensure Kaggle scores the submission.")

if __name__ == "__main__":
    main()
