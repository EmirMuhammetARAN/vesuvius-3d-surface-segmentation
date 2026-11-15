"""
Complete Training Pipeline for Vesuvius Challenge Surface Detection
Using 3D ResEncUNet, Mixed Precision (AMP), Cosine Annealing,
and 1st-Place 5-step Topological Post-Processing Evaluation.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import argparse
import time
import math
import ctypes
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import Vesuvius3DPatchDataset
from src.models.unet3d import ResEncUNet3D
from src.models.segresnet3d import PureSegResNet3D
from src.losses import VesuviusCompoundLoss
from src.postprocess import full_postprocess_pipeline
from src.metrics import compute_competition_metric


def get_args():
    parser = argparse.ArgumentParser(description="Train 3D ResEncUNet for Vesuvius Challenge")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2, 1st place setting)")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size cube (128 or 192)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--samples_per_volume", type=int, default=4, help="Random crops per volume per epoch")
    parser.add_argument("--val_scroll_id", type=int, default=26010, help="Scroll ID to hold out for validation")
    parser.add_argument("--all_data", action="store_true", help="Train on all data without validation split")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5 epochs)")
    parser.add_argument("--lr_cycle_start", type=int, default=1, help="Epoch to start/restart cosine decay cycle")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights to initialize model")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--pos_weight", type=float, default=2.5, help="Positive class weight for BCE loss")
    parser.add_argument("--save_interval", type=int, default=25, help="Save snapshot checkpoint every N epochs")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for checkpoint naming (e.g. v2)")
    parser.add_argument("--arch", type=str, default="resenc", choices=["resenc", "segresnet"], help="Model architecture: resenc or segresnet")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps: int = 1, epoch: int = 1, total_epochs: int = 160, start_time: float = 0.0):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    optimizer.zero_grad()
    
    n_batches = len(loader)
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss_dict = criterion(logits, labels)
            loss = loss_dict["loss"] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
            try:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except AssertionError:
                scaler.update()
                optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * accum_steps)
        running_bce += loss_dict["bce"].item()
        running_dice += loss_dict["dice"].item()

        # CANLI İLERLEME (Her 40 adımda bir = ~10-15 saniyede bir ekranı canlı güncelle)
        if (step + 1) % 40 == 0 or (step + 1) == n_batches or step == 0:
            step_num = step + 1
            pct_epoch = (step_num / n_batches) * 100.0
            overall_done = (epoch - 1) * n_batches + step_num
            total_all_steps = total_epochs * n_batches
            pct_overall = (overall_done / total_all_steps) * 100.0
            
            elapsed = time.time() - start_time if start_time > 0 else 1.0
            eta_sec = (elapsed / max(1, overall_done)) * (total_all_steps - overall_done)
            eta_h = eta_sec / 3600.0
            cur_loss = (loss.item() * accum_steps)
            cur_bce = loss_dict["bce"].item()
            cur_dice = loss_dict["dice"].item()
            
            bar_len = 20
            filled = min(bar_len, int(bar_len * (pct_overall / 100.0)))
            bar_str = "=" * filled + ">" + " " * max(0, bar_len - filled - 1) if filled < bar_len else "=" * bar_len
            
            print(
                f"[{bar_str}] GENEL: %{pct_overall:5.1f} | Ep {epoch:03d}/{total_epochs:03d} (Adım {step_num:03d}/{n_batches} - %{pct_epoch:3.0f}) | "
                f"Loss: {cur_loss:.4f} (BCE: {cur_bce:.4f}, Dice: {cur_dice:.4f}) | Kalan: {eta_h:.1f} sa",
                flush=True
            )

    return running_loss / n_batches, running_bce / n_batches, running_dice / n_batches


@torch.no_grad()
def validate(model, loader, device, max_val_steps: int = 60):
    """
    High-speed GPU 3D Dice validation during training.
    Computes exact volumetric Dice in seconds directly on GPU,
    avoiding the massive CPU KD-Tree and topological bottleneck.
    """
    model.eval()
    dice_scores = []

    pbar = tqdm(loader, desc="Validating", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)

        preds = (probs >= 0.35).float()
        targets = (labels == 1).float()

        intersection = (preds * targets).sum(dim=(-3, -2, -1))
        union = preds.sum(dim=(-3, -2, -1)) + targets.sum(dim=(-3, -2, -1))
        dice = (2.0 * intersection) / (union + 1e-6)

        dice_scores.extend(dice.cpu().tolist())
        pbar.set_postfix({"val_dice": f"{np.mean(dice_scores):.4f}"})

        if max_val_steps and (step + 1) >= max_val_steps:
            break

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    return mean_dice, mean_dice


def set_high_priority_and_prevent_sleep():
    """Sets process priority to HIGH on Windows and prevents sleep/standby during training."""
    if sys.platform == "win32":
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("[+] Process Priority set to: HIGH (Windows background throttling disabled)")
        except Exception as e:
            print(f"[!] Warning: Could not set high priority: {e}")

        try:
            # ES_CONTINUOUS = 0x80000000, ES_SYSTEM_REQUIRED = 0x00000001, ES_AWAYMODE_REQUIRED = 0x00000040
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000040
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
            print("[+] Windows Sleep Prevention: Active (system sleep/standby disabled during training)")
        except Exception as e:
            print(f"[!] Warning: Could not set sleep prevention: {e}")


def get_cosine_lr(epoch: int, total_epochs: int, base_lr: float, eta_min: float = 1e-6, cycle_start: int = 1) -> float:
    """Computes exact cosine annealing learning rate for epoch (1-indexed) with cycle_start support."""
    curr = max(cycle_start, epoch)
    ratio = (curr - cycle_start) / max(1, total_epochs - cycle_start)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * ratio))


def main():
    set_high_priority_and_prevent_sleep()
    
    # Maximum PyTorch GPU Hardware Optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("[+] PyTorch Optimizations: cuDNN Benchmark=True, Float32 MatMul='high'")

    args = get_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load metadata
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    print(f"[*] Total volumes available: {len(df)}")

    if args.all_data:
        train_df = df
        val_df = None
        print("[*] Mode: Training on all data without validation split")
    else:
        # GroupKFold split on scroll_id
        val_df = df[df["scroll_id"] == args.val_scroll_id].reset_index(drop=True)
        train_df = df[df["scroll_id"] != args.val_scroll_id].reset_index(drop=True)
        print(f"[*] Train volumes: {len(train_df)} | Held-out Val volumes (Scroll {args.val_scroll_id}): {len(val_df)}")

    patch_size_cube = (args.patch_size, args.patch_size, args.patch_size)

    # Dataset & Loaders
    train_ds = Vesuvius3DPatchDataset(
        train_df,
        data_dir=args.data_dir,
        patch_size=patch_size_cube,
        samples_per_volume=args.samples_per_volume,
        fg_oversample_prob=0.67,
        is_train=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_df is not None and len(val_df) > 0:
        val_ds = Vesuvius3DPatchDataset(
            val_df,
            data_dir=args.data_dir,
            patch_size=patch_size_cube,
            samples_per_volume=2,
            is_train=False
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    if args.arch == "segresnet":
        try:
            from monai.networks.nets import SegResNet
            model = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            ).to(device)
        except ImportError:
            model = PureSegResNet3D(
                in_channels=1,
                out_channels=1,
                init_filters=32,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
            ).to(device)
        arch_name = "SegResNet3D"
        arch_prefix = "segresnet3d"
    else:
        model = ResEncUNet3D(in_channels=1, out_channels=1).to(device)
        arch_name = "ResEncUNet3D"
        arch_prefix = "resenc_unet3d"

    print(f"[*] {arch_name} initialized with ~{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Load pretrained weights if specified (e.g. from 128 patch training)
    if args.pretrained and Path(args.pretrained).exists():
        print(f"[*] Loading pretrained weights from: {args.pretrained}")
        weights = torch.load(args.pretrained, map_location=device)
        if isinstance(weights, dict) and "model_state_dict" in weights:
            model.load_state_dict(weights["model_state_dict"])
            print(f"[+] Loaded model_state_dict from checkpoint (Score: {weights.get('score', weights.get('best_score', 'N/A'))})")
        else:
            model.load_state_dict(weights)
            print(f"[+] Successfully loaded pretrained weights from {args.pretrained}")

    criterion = VesuviusCompoundLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=args.pos_weight, ignore_label=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_score = -float('inf') if args.all_data else 0.0
    best_epoch = 1
    no_improve_count = 0
    ckpt_prefix = f"{arch_prefix}_p{args.patch_size}_{args.tag}" if args.tag else f"{arch_prefix}_p{args.patch_size}"
    latest_checkpoint = save_dir / f"{ckpt_prefix}_latest.pth"

    # Auto-resume or resume if requested
    if args.resume and latest_checkpoint.exists():
        print(f"[*] Resuming from checkpoint: {latest_checkpoint}")
        ckpt = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", ckpt.get("score", 0.0))
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        no_improve_count = ckpt.get("no_improve_count", 0)
        resumed_lr = get_cosine_lr(start_epoch, args.epochs, args.lr, cycle_start=args.lr_cycle_start)
        for pg in optimizer.param_groups:
            pg['lr'] = resumed_lr
        print(f"[+] Successfully resumed from Epoch {start_epoch} (Best Score: {best_score:.4f} at Ep {best_epoch}, Resumed LR: {resumed_lr:.2e})")

    print(f"\n[*] Starting training from Epoch {start_epoch} to {args.epochs} (Batch Size: {args.batch_size}, Patch: {args.patch_size}^3, Patience: {args.patience})...", flush=True)
    training_start_time = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            current_lr = get_cosine_lr(epoch, args.epochs, args.lr, cycle_start=args.lr_cycle_start)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            train_loss, train_bce, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                accum_steps=args.accum_steps,
                epoch=epoch,
                total_epochs=args.epochs,
                start_time=training_start_time
            )
            elapsed = time.time() - t0
            elapsed_total = time.time() - training_start_time
            epochs_done = epoch - start_epoch + 1
            eta_sec = (elapsed_total / max(1, epochs_done)) * (args.epochs - epoch)
            eta_h = eta_sec / 3600.0
            pct_overall = (epoch / args.epochs) * 100.0

            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            
            print(f"\n{'='*80}", flush=True)
            print(f"[>>>] EPOCH {epoch:03d}/{args.epochs:03d} BİTTİ | GENEL: %{pct_overall:5.1f} | Süre: {elapsed:.1f}s | Toplam Geçen: {elapsed_total/3600:.2f} sa | Kalan: {eta_h:.2f} sa", flush=True)
            print(f"      Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice:.4f}) | LR: {current_lr:.2e} | VRAM: {vram_gb:.1f}GB", flush=True)
            print(f"{'='*80}\n", flush=True)

            if val_loader is not None:
                raw_score, post_score = validate(model, val_loader, device)
                print(f"[*] Val Raw Score: {raw_score:.4f} | Postprocessed Score: {post_score:.4f}", flush=True)
                current_eval = post_score
            else:
                current_eval = -train_loss

            if current_eval > best_score:
                best_score = current_eval
                best_epoch = epoch
                no_improve_count = 0
                best_path = save_dir / f"{ckpt_prefix}_best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"    --> [BEST] Model saved to {best_path} (Score: {best_score:.4f})")
            else:
                no_improve_count += 1
                print(f"    --> [NO IMPROVE] Patience: {no_improve_count}/{args.patience} (Best: {best_score:.4f} at Epoch {best_epoch})")
                if no_improve_count >= args.patience:
                    print(f"\n[*] Early Stopping triggered after {no_improve_count} epochs without improvement!")
                    print(f"[*] Best Model locked at Epoch {best_epoch} with Score: {best_score:.4f}")
                    torch.save({
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "score": current_eval,
                        "best_score": best_score,
                        "no_improve_count": no_improve_count
                    }, latest_checkpoint)
                    break

            # Save interval snapshot if requested
            if args.save_interval > 0 and epoch % args.save_interval == 0:
                interval_path = save_dir / f"{ckpt_prefix}_epoch{epoch}.pth"
                torch.save(model.state_dict(), interval_path)
                print(f"    --> [SNAPSHOT] Interval model saved to {interval_path}")

            # Save full checkpoint
            torch.save({
                "epoch": epoch,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "score": current_eval,
                "best_score": best_score,
                "no_improve_count": no_improve_count
            }, latest_checkpoint)

        final_path = save_dir / f"{ckpt_prefix}_final.pth"
        torch.save(model.state_dict(), final_path)
        print(f"\n[+] Training loop ended! Best Score: {best_score:.4f} at Epoch {best_epoch}. Final model saved to {final_path}.")

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected! Saving emergency checkpoint before exit...")
        emergency_path = save_dir / f"{ckpt_prefix}_interrupted.pth"
        save_dict = {
            "epoch": epoch if 'epoch' in locals() else start_epoch,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score
        }
        torch.save(save_dict, emergency_path)
        torch.save(save_dict, latest_checkpoint)
        print(f"[+] Emergency checkpoint safely saved to: {latest_checkpoint}")
        print("[+] You can resume anytime by running: python train.py --resume")
    finally:
        if sys.platform == "win32":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass


if __name__ == "__main__":
    main()

