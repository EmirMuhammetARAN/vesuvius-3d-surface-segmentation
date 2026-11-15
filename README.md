# Vesuvius Challenge — 3D Papyrus Surface Segmentation

End-to-end 3D volumetric segmentation pipeline built from scratch in **PyTorch** and **SciPy/scikit-image** for the [Kaggle Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) competition.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Models](https://img.shields.io/badge/HuggingFace-Model%20Weights-yellow.svg)](https://huggingface.co/emiraran/vesuvius-3d-surface-segmentation)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Vesuvius%20Surface%20Detection-20BEFF.svg)](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)

> **Project Evolution Note:**  
> This repository documents the complete rebuild from the initial **v1 TensorFlow/Keras baseline** ([`archive/v1_model.ipynb`](./archive/v1_model.ipynb), `96^3` patches, **0.40420** Private LB) to the **v2 modular PyTorch pipeline** (`src/`, `train.py`, `notebooks/kaggle_inference.py`), which scales progressive `160^3 -> 192^3` 3D backbones, 10-model multi-architecture ensembling, and topological surface repair to **0.54788** Private LB (**+0.14368** over v1).

---

## Pretrained Ensemble Checkpoints (Hugging Face)

All 10 trained 3D checkpoints (`PureSegResNet3D` & `ResEncUNet3D`, ~694 MB total) are hosted on Hugging Face Hub:

- **Model Repository:** [emiraran/vesuvius-3d-surface-segmentation](https://huggingface.co/emiraran/vesuvius-3d-surface-segmentation)

```python
import torch
from huggingface_hub import hf_hub_download
from src.models import PureSegResNet3D

ckpt_path = hf_hub_download(
    repo_id="emiraran/vesuvius-3d-surface-segmentation",
    filename="weights/segresnet3d_p192_finetune192_best.pth",
)
model = PureSegResNet3D(
    in_channels=1, out_channels=1, init_filters=32,
    blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
)
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(state.get("model_state_dict", state))
model.eval()
```

---

## Competition Progression (Official Kaggle Evaluations)

| Milestone | Pipeline & Training Setup | Public LB | Private LB | Delta (vs. v1) |
|---|---|---:|---:|---:|
| **v1 (`archive/v1_model.ipynb`)** | TensorFlow 3D U-Net (`96^3`, 4 diagonal corner samples/vol) | `0.39477` | `0.40420` | — |
| **v2 Initial PyTorch (`2026-09-15`)** | `ResEncUNet3D` (`128^3`), ignore-mask compound loss | `0.46164` | `0.47161` | `+0.06741` |
| **v2 Progressive Scale (`2026-09-27`)** | `160^3` / `192^3` models + 3D sliding-window TTA + morphology | `0.52229` | `0.52719` | `+0.12299` |
| **v2 8-Model Ensemble (`2026-11-01`)** | 3x `SegResNet3D` + 5x `ResEncUNet3D` (`160^3` + `192^3`) | `0.52601` | `0.53265` | `+0.12845` |
| **v2 10-Model Ensemble (`2026-11-14`, K23)** | + `192^3` all-data `SegResNet3D` fine-tune (150 epochs on L4) | **`0.54030`** | **`0.54788`** | **`+0.14368`** |

---

## Architecture & Pipeline Design

### 1. Dual-Backbone 10-Model 3D Ensemble

The final Kaggle inference notebook ([`notebooks/kaggle_inference.py`](./notebooks/kaggle_inference.py)) dynamically loads and blends 10 standalone PyTorch 3D weights:

| # | Checkpoint | Backbone | Parameters | Patch Size | Overlap | TTA | Weight | Share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `segresnet3d_p192_finetune192_best.pth` | `PureSegResNet3D` | `18,795,969` | `192^3` | `0.665` | `2x` | `4.5` | `19.1%` |
| 2 | `segresnet3d_p192_finetune192_epoch150.pth` | `PureSegResNet3D` | `18,795,969` | `192^3` | `0.665` | `2x` | `2.5` | `10.6%` |
| 3 | `segresnet3d_p160_fold2_best.pth` | `PureSegResNet3D` | `18,795,969` | `160^3` | `0.500` | `4x` | `3.5` | `14.9%` |
| 4 | `segresnet3d_p160_best.pth` | `PureSegResNet3D` | `18,795,969` | `160^3` | `0.500` | `4x` | `3.0` | `12.8%` |
| 5 | `segresnet3d_p192_colab_fold3_best.pth` | `PureSegResNet3D` | `18,795,969` | `192^3` | `0.665` | `2x` | `2.5` | `10.6%` |
| 6 | `resenc_unet3d_p192_alldata_best.pth` | `ResEncUNet3D` | `15,912,097` | `192^3` | `0.665` | `2x` | `2.0` | `8.5%` |
| 7 | `resenc_unet3d_p192_colab_fold3_best.pth` | `ResEncUNet3D` | `15,912,097` | `192^3` | `0.665` | `2x` | `2.0` | `8.5%` |
| 8 | `resenc_unet3d_p192_v2_best.pth` | `ResEncUNet3D` | `15,912,097` | `192^3` | `0.665` | `2x` | `1.5` | `6.4%` |
| 9 | `resenc_unet3d_p192_fold2_best.pth` | `ResEncUNet3D` | `15,912,097` | `192^3` | `0.665` | `2x` | `1.0` | `4.3%` |
| 10 | `resenc_unet3d_p160_p160_best.pth` | `ResEncUNet3D` | `15,912,097` | `160^3` | `0.500` | `2x` | `1.0` | `4.3%` |

- **`PureSegResNet3D`** ([`src/models/segresnet3d.py`](./src/models/segresnet3d.py)): Pure-PyTorch implementation bit-for-bit compatible with MONAI `SegResNet` state dicts (`GroupNorm(8, C)`, residual blocks `[1, 2, 2, 4]`, trilinear upsampling), requiring zero external MONAI dependencies during Kaggle offline inference.
- **`ResEncUNet3D`** ([`src/models/unet3d.py`](./src/models/unet3d.py)): 5-stage 3D U-Net with residual encoder blocks (`[32, 64, 128, 256, 320]`), `InstanceNorm3d`, `LeakyReLU(0.01)`, and transposed convolution decoder blocks.

### 2. Training Pipeline ([`train.py`](./train.py))
- **Foreground Oversampling (`src/dataset.py`):** 67% probability of sampling 3D patches centered around annotated papyrus voxels (`label == 1`), plus 3D flip and `90°` spatial rotation augmentations.
- **Ignore-Aware Compound Loss (`src/losses.py`):** Masks out unlabeled CT boundaries (`label == 2`) cleanly during both `BCEWithLogits(pos_weight=2.5)` and `SoftDiceLoss` computation (`0.5 * BCE + 0.5 * Dice`).
- **Training Stability:** Mixed Precision (`torch.amp.autocast`), gradient accumulation (`--accum_steps`), gradient norm clipping (`max_norm=1.0`), `AdamW` + cosine annealing LR schedule, and interrupt/resume checkpointing.

### 3. Inference & Topological Post-Processing (`notebooks/kaggle_inference.py` & `src/postprocess.py`)
- **Gaussian-Blended Sliding Window:** 3D Gaussian weight map eliminates tile seam artifacts across overlapping `160^3` and `192^3` patches, with automatic OOM batch-size fallback (`2 -> 1`).
- **Adaptive 9-Hour Kaggle Pace Controller:** Automatically scales TTA/overlap across Tiers 1–4 depending on test-set volume count so hidden evaluation never times out.
- **Seeded Hysteresis & Topological Repair (`seeded_hysteresis_postprocess`):**
  1. Seeded 3D hysteresis propagation (`T_high=0.66`, `T_low=0.42`).
  2. Anisotropic morphological closing (`z_radius=3`, `xy_radius=2`) to reconnect cracked sheet planes prior to component filtering.
  3. Connected-component pruning (`dust_min_size=10000` voxels) and rim-subsampled 2D heightmap linear interpolation per sheet (`patch_large_holes_heightmap_fast`) to repair internal sheet tears.
  4. 6/26-connected watertight tiny-hole plugging (`plug_watertight_holes_6c`) and 3D cavity fill (`binary_fill_holes`).

---

## Technical Retrospective (vs. 1st Place `0.62702`)

Comparing our custom PyTorch/Colab L4 run (`0.54788`) against the competition's [1st Place Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su) highlights three concrete architectural and compute gaps:
1. **Training Scale & Framework (`nnU-Net v2`):** Top solutions trained standard `nnU-Net v2` pipelines for `4,000` epochs (`250` batches/epoch = `2,000,000` patches) on `128^3` followed by `192^3`/`256^3` fine-tuning with deep supervision and heavy 3D augmentations (elastic deformation, gamma, blur, noise) — roughly `~17x` more training patches than our 150-epoch L4 fine-tune (`117,900` patches).
2. **Threshold vs. Hysteresis Calibration:** While hysteresis (`0.66 / 0.42`) suppresses weak background fragments, direct calibrated thresholding (`0.26–0.40` post-ensemble) combined with stronger single-model calibration preserves more borderline surface connectivity.
3. **Post-Processing Guardrails:** The 1st-place solution validated topological hole counts before and after heightmap interpolation (reverting any patch on highly curved scrolls that increased hole count) and used a 256-entry `2x2x2` Euler-style LUT rather than heuristic neighbor counts.

---

## Repository Structure

```text
vesuvius-3d-surface-segmentation/
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # Vesuvius3DPatchDataset (foreground oversampling + 3D aug)
│   ├── losses.py                    # VesuviusCompoundLoss (BCE + SoftDice ignoring label==2)
│   ├── metrics.py                   # Official competition proxy (SurfaceDice, VOI, TopoScore)
│   ├── postprocess.py               # 5-step topological post-processing pipeline
│   └── models/
│       ├── __init__.py
│       ├── segresnet3d.py           # PureSegResNet3D (MONAI-compatible, zero external deps)
│       └── unet3d.py                # ResEncUNet3D (nnU-Net-style 3D residual encoder U-Net)
├── scripts/
│   ├── download_data.py             # Kaggle competition dataset downloader via kagglehub
│   └── make_submission.py           # Local ensemble validation & submission.zip generator
├── notebooks/
│   ├── kaggle_inference.py          # Self-contained offline Kaggle inference + 4-tier pacing
│   ├── colab_finetune_192.ipynb     # Google Colab L4 192^3 all-data fine-tuning notebook
│   └── vesuvius_kernel/
│       ├── vesuvius.ipynb           # Synced Kaggle submission kernel (Version 23)
│       └── kernel-metadata.json
├── archive/
│   └── v1_model.ipynb               # Archived March 2026 TensorFlow/Keras baseline (0.40420)
├── tests/
│   ├── test_metrics.py              # Unit tests for SurfaceDice, VOI split/merge, TopoScore
│   ├── test_pipeline.py             # Forward shape tests (ResEncUNet3D & PureSegResNet3D) + loss
│   └── test_postprocess.py          # Hole plugging, cavity fill & pipeline integration tests
├── train.py                         # Full CLI training & fine-tuning entrypoint
├── requirements.txt
├── LICENSE                          # MIT License
└── README.md
```

---

## Quickstart & Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests (metrics, architectures, losses, post-processing)
pytest -v tests/

# 3. Fine-tune 192^3 SegResNet3D on all volumes (requires Kaggle competition dataset in dataset/)
python train.py --arch segresnet --patch_size 192 --batch_size 1 --accum_steps 2 --all_data --epochs 160 --lr 1e-4
```

## License

MIT License — see [`LICENSE`](./LICENSE).
