"""
Local 3D Ensemble Inference & Kaggle Submission Pipeline.
Features:
- Multi-model ensembling (128^3 + 192^3 models blended)
- 4x 3D Test-Time Augmentation (D, H, W flips)
- 3D sliding window with Gaussian blending overlap
- 5-step topological post-processing
"""

import os
import zipfile
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import torch
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.unet3d import ResEncUNet3D
from src.postprocess import full_postprocess_pipeline


def gaussian_weight_map_3d(patch_size: tuple, sigma_scale: float = 0.125) -> np.ndarray:
    """Creates a 3D Gaussian weight map to eliminate boundary seams during sliding window stitching."""
    D, H, W = patch_size
    sigmas = (D * sigma_scale, H * sigma_scale, W * sigma_scale)
    
    z = np.exp(-0.5 * ((np.arange(D) - D / 2.0) / sigmas[0]) ** 2)
    y = np.exp(-0.5 * ((np.arange(H) - H / 2.0) / sigmas[1]) ** 2)
    x = np.exp(-0.5 * ((np.arange(W) - W / 2.0) / sigmas[2]) ** 2)
    
    weights = z[:, None, None] * y[None, :, None] * x[None, None, :]
    weights /= (weights.max() + 1e-8)
    return weights.astype(np.float32)


@torch.no_grad()
def predict_patch(model: torch.nn.Module, tensor_patch: torch.Tensor, use_tta: bool = False) -> np.ndarray:
    """Predicts a 3D patch with optional 4x Test-Time Augmentation (flips across D, H, W)."""
    with torch.amp.autocast("cuda"):
        logits = model(tensor_patch)
        if use_tta:
            for dim in (2, 3, 4):
                logits = logits + torch.flip(model(torch.flip(tensor_patch, dims=[dim])), dims=[dim])
            logits = logits / 4.0
        probs = torch.sigmoid(logits)
    return probs.squeeze().cpu().numpy()


@torch.no_grad()
def predict_volume_sliding_window(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: tuple = (128, 128, 128),
    overlap: float = 0.5,
    use_tta: bool = False,
    device: str = "cuda"
) -> np.ndarray:
    """
    Performs 3D sliding window inference on full volume with Gaussian blending and optional TTA.
    """
    model.eval()
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    step_d = max(1, int(pD * (1.0 - overlap)))
    step_h = max(1, int(pH * (1.0 - overlap)))
    step_w = max(1, int(pW * (1.0 - overlap)))

    # Pad if volume is smaller than patch size
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    pD_vol, pH_vol, pW_vol = volume.shape
    prob_accum = np.zeros((pD_vol, pH_vol, pW_vol), dtype=np.float32)
    weight_accum = np.zeros((pD_vol, pH_vol, pW_vol), dtype=np.float32)
    gaussian_weights = gaussian_weight_map_3d(patch_size)

    # Normalize volume
    norm_vol = volume.astype(np.float32) / 255.0

    z_starts = list(range(0, pD_vol - pD + 1, step_d))
    if z_starts[-1] != pD_vol - pD:
        z_starts.append(pD_vol - pD)

    y_starts = list(range(0, pH_vol - pH + 1, step_h))
    if y_starts[-1] != pH_vol - pH:
        y_starts.append(pH_vol - pH)

    x_starts = list(range(0, pW_vol - pW + 1, step_w))
    if x_starts[-1] != pW_vol - pW:
        x_starts.append(pW_vol - pW)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    pbar = tqdm(total=total_patches, desc=f"Sliding Window {patch_size[0]}^3 (TTA={use_tta})", leave=False)

    for zs in z_starts:
        for ys in y_starts:
            for xs in x_starts:
                patch = norm_vol[zs : zs + pD, ys : ys + pH, xs : xs + pW]
                tensor_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                probs = predict_patch(model, tensor_patch, use_tta=use_tta)

                prob_accum[zs : zs + pD, ys : ys + pH, xs : xs + pW] += probs * gaussian_weights
                weight_accum[zs : zs + pD, ys : ys + pH, xs : xs + pW] += gaussian_weights
                pbar.update(1)

    pbar.close()
    final_probs = prob_accum / np.maximum(weight_accum, 1e-6)
    final_probs = final_probs[:D, :H, :W]
    return final_probs


def load_model(weights_path: str, device: str) -> torch.nn.Module:
    """Loads ResEncUNet3D and restores weights cleanly."""
    model = ResEncUNet3D(in_channels=1, out_channels=1).to(device)
    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location=device)
        state_dict = weights.get("model_state_dict", weights)
        model.load_state_dict(state_dict)
        score = weights.get("score", weights.get("best_score", "N/A"))
        print(f"[+] Loaded weights from {weights_path} (Validation Score: {score})")
    else:
        print(f"[!] Warning: Model weights {weights_path} not found.")
    model.eval()
    return model


def generate_submission(
    model_128_path: str = "checkpoints/resenc_unet3d_p128_best.pth",
    model_192_path: str = "checkpoints/resenc_unet3d_p192_best.pth",
    use_ensemble: bool = True,
    use_tta: bool = True,
    data_dir: str = "dataset",
    output_dir: str = "submissions",
    threshold: float = 0.35,
    overlap: float = 0.5,
    min_component: int = 20000
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    models_to_run = []
    # 1. Check 128 model
    if os.path.exists(model_128_path):
        m128 = load_model(model_128_path, device)
        models_to_run.append((m128, (128, 128, 128), "ResEncUNet3D-p128"))

    # 2. Check 192 model
    if os.path.exists(model_192_path) and use_ensemble:
        m192 = load_model(model_192_path, device)
        models_to_run.append((m192, (192, 192, 192), "ResEncUNet3D-p192"))

    if not models_to_run:
        print("[!] Error: No model checkpoints found to run inference!")
        return None

    print(f"[*] Ensemble active: {len(models_to_run)} model(s) will be blended {[name for _, _, name in models_to_run]}")
    print(f"[*] Test-Time Augmentation (TTA): {'Active (4x flips)' if use_tta else 'Off'}")
    print(f"[*] Sliding Window Overlap: {int(overlap * 100)}% with Gaussian Blending")

    test_csv_path = os.path.join(data_dir, "test.csv")
    test_img_dir = os.path.join(data_dir, "test_images")
    df_test = pd.read_csv(test_csv_path)

    print(f"[*] Processing {len(df_test)} test volume(s)...")
    generated_tifs = []

    for _, row in df_test.iterrows():
        img_id = str(row["id"])
        img_path = os.path.join(test_img_dir, f"{img_id}.tif")
        if not os.path.exists(img_path):
            print(f"[!] Skipping missing volume: {img_path}")
            continue

        print(f"\n[*] Processing test volume {img_id}...")
        volume = tifffile.imread(img_path)
        print(f"    Dimensions: {volume.shape} (D, H, W)")

        # Ensemble accumulator
        ensemble_prob_map = np.zeros(volume.shape, dtype=np.float32)

        for model, patch_size, name in models_to_run:
            print(f"    Running 3D sliding window for {name} ({patch_size[0]}^3)...")
            prob_map = predict_volume_sliding_window(
                model,
                volume,
                patch_size=patch_size,
                overlap=overlap,
                use_tta=use_tta,
                device=device
            )
            ensemble_prob_map += (prob_map / len(models_to_run))

        print(f"    Applying topological post-processing (threshold={threshold})...")
        binary_mask = full_postprocess_pipeline(
            ensemble_prob_map,
            threshold=threshold,
            min_component_voxels=min_component,
            closing_radius=3,
            use_heightmap=True
        )

        out_tif_path = out_path / f"{img_id}.tif"
        tifffile.imwrite(str(out_tif_path), binary_mask.astype(np.uint8))
        generated_tifs.append(out_tif_path)
        print(f"[+] Saved prediction mask to {out_tif_path}")

    # Package into submission.zip
    zip_path = out_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for tif in generated_tifs:
            zf.write(tif, arcname=tif.name)

    print(f"\n[+] SUCCESS! Created valid submission file at: {zip_path}")
    print(f"    Zip contents: {[t.name for t in generated_tifs]}")
    return str(zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission with Ensemble & TTA")
    parser.add_argument("--model_128", type=str, default="checkpoints/resenc_unet3d_p128_best.pth", help="Path to 128 model")
    parser.add_argument("--model_192", type=str, default="checkpoints/resenc_unet3d_p192_best.pth", help="Path to 192 model")
    parser.add_argument("--no_ensemble", action="store_true", help="Disable ensemble and use single model")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--threshold", type=float, default=0.35, help="Binarization threshold")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap (default: 0.5 = 50 percent)")
    parser.add_argument("--min_component", type=int, default=20000, help="Min component voxels")
    args = parser.parse_args()

    generate_submission(
        model_128_path=args.model_128,
        model_192_path=args.model_192,
        use_ensemble=not args.no_ensemble,
        use_tta=not args.no_tta,
        threshold=args.threshold,
        overlap=args.overlap,
        min_component=args.min_component
    )
