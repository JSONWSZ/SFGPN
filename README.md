# Soft Foreground-Guided Cross-Scale Perception for Visible–Infrared Person Re-Identification

[![DOI](https://zenodo.org/badge/1142421501.svg)](https://doi.org/10.5281/zenodo.18376063)

> **Note to Readers and Reviewers:**
> This repository is the official implementation of the paper **"Soft Foreground-Guided Cross-Scale Perception for Visible–Infrared Person Re-Identification"** (submitted to *The Visual Computer*). The source code, trained models, generated masks, configuration files, and evaluation scripts are permanently hosted to ensure full transparency and reproducibility of all experiments presented in the paper. The repository is also archived at [Zenodo](https://doi.org/10.5281/zenodo.18376063).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Reproducing Paper Results (Tables 1–9)](#5-reproducing-paper-results-tables-19)
   - [Table 1 & 2: SOTA Comparison (SYSU-MM01, RegDB, LLCM)](#table-1--2-state-of-the-art-comparison)
   - [Table 3: Ablation Study](#table-3-ablation-study)
   - [Tables 4 & 5: FRL/CSFP Insertion Position](#tables-4--5-frlcsfp-insertion-position)
   - [Table 6: CSFP Branch Loss Strategy](#table-6-csfp-branch-loss-strategy)
   - [Table 7 & Fig. 6: Hyperparameter Analysis](#table-7--fig-6-hyperparameter-analysis)
   - [Table 8: Loss Introduction Schedule](#table-8-loss-introduction-schedule)
   - [Table 9: Weighted Identity Center Ablation](#table-9-weighted-identity-center-ablation)
   - [Table 10: SCG Loss Decomposition](#table-10-scg-loss-decomposition)
   - [Table 11: FRL Heatmap Quantitative Evaluation](#table-11-frl-heatmap-quantitative-evaluation)
   - [Table 12: Mask Noise Robustness](#table-12-mask-noise-robustness)
   - [Table 13: Reusability (Plug-and-Play)](#table-13-reusability-plug-and-play)
   - [Table 14: Model Complexity Analysis](#table-14-model-complexity-analysis)
6. [Training](#6-training)
7. [Testing](#7-testing)
8. [Visualization & Analysis Scripts](#8-visualization--analysis-scripts)
9. [Pretrained Models & Processed Data](#9-pretrained-models--processed-data)
10. [Citation](#10-citation)
11. [Acknowledgments](#11-acknowledgments)

---

## 1. Overview

Visible–Infrared Person Re-Identification (VI-ReID) is challenged by significant modality discrepancy inherent in background regions. We propose the **Soft Foreground Guided Perception Network (SFGPN)**, which combines:

- **Foreground Region Localization (FRL)** module — generates pixel-level attention heatmaps to localize pedestrian body regions.
- **Soft Consistency Guidance (SCG)** loss — treats externally generated body masks as soft supervisory signals rather than rigid feature filters, providing robustness to segmentation noise.
- **Cross-Scale Foreground Perception (CSFP)** module — captures complementary multi-scale attributes (local details + global semantics) within localized foreground regions.
- **Weighted Hetero-Center (WHC) and Weighted Center (WC) losses** — cosine-similarity-weighted identity centers for robust intra-class compactness across modalities.

**Key practical advantage:** FRL operates exclusively during training and is discarded at inference — introducing **zero additional inference cost**.

### Key Implementation Files

| Component | File | Line/Class |
|---|---|---|
| FRL Module | [`SFGPN_noise/model.py`](SFGPN_noise/model.py) | `FocusModule` (line 146) |
| CSFP Module | [`SFGPN_noise/model.py`](SFGPN_noise/model.py) | `MultiScaleModule` (line 214) |
| SCG Loss (L2 + Dice) | [`SFGPN_noise/loss.py`](SFGPN_noise/loss.py) | `FocalLoss` (line 144) |
| WHC & WC Losses | [`SFGPN_noise/loss.py`](SFGPN_noise/loss.py) | `modality_center_loss` (line 173) |
| BGS & BRE Augmentations | [`SFGPN_noise/data_loader.py`](SFGPN_noise/data_loader.py) | `regional_random_grayscale` / `regional_random_erasing` |

---

## 2. Repository Structure

```
SFGPN/
├── README.md                          # This file
├── eval_frl_heatmap.py                # FRL heatmap quantitative evaluation
│
├── SFGPN_noise/                       # Main SFGPN implementation (noise-robust version)
│   ├── train.py                       # Training script (all datasets)
│   ├── test_matrix.py                 # Testing/evaluation script
│   ├── model.py                       # embed_net, FocusModule (FRL), MultiScaleModule (CSFP)
│   ├── loss.py                        # SCG loss, WHC/WC loss, TripletLoss_WRT
│   ├── data_loader.py                 # Data loading with BGS, BRE, mask noise
│   ├── data_manager.py                # Dataset split & query/gallery processing
│   ├── eval_metrics.py                # CMC, mAP, mINP evaluation
│   ├── resnet.py                      # ResNet-50 backbone
│   ├── utils.py                       # Logger, AverageMeter, IdentitySampler
│   ├── random_erasing.py              # Random Erasing augmentation
│   ├── pre_process_sysu_mask.py       # SYSU-MM01 data preprocessing
│   ├── requirements.txt               # Python dependencies
│   ├── train_regdb.sh                 # RegDB training shell script
│   │
│   ├── PedestrianSegmentation/
│   │   └── Segmentation.py            # YOLO11 instance segmentation for mask generation
│   │
│   ├── FLOPs_Params.py                # FLOPs & parameter count measurement
│   ├── Inference_Time.py              # Inference time & FPS measurement
│   ├── Grad-CAM.py                    # Multi-branch Grad-CAM visualization
│   ├── Grad-CAM-plusplus.py           # Grad-CAM++ visualization
│   ├── t-SNE.py                       # t-SNE feature visualization
│   ├── plot_intra_inter.py            # Intra/inter-class distance distribution
│   ├── test_query_SYSU.py             # SYSU-MM01 retrieval result visualization
│   ├── test_query_RegDB.py            # RegDB retrieval result visualization
│   ├── test_heatmap_RE.py             # FRL heatmap + random erasing visualization
│   ├── test_mask.py                   # Mask & heatmap overlay visualization
│   └── bar.py                         # Bar chart visualization
│
├── SFGPN_HMV_noise/                   # HMV (Hard Mask Variant) for comparison experiments
│   └── ...                            # Same structure as SFGPN_noise/
│
├── AGW/                               # AGW baseline (for reusability experiments)
├── MMN/                               # MMN baseline (for reusability experiments)
└── MCLNet/                            # MCLNet baseline (for reusability experiments)
```

---

## 3. Environment Setup

### Requirements

All experiments were conducted with the following environment:

- **OS**: Ubuntu 20.04 / Windows 10
- **Python**: 3.10+
- **CUDA**: 12.1
- **GPU**: NVIDIA RTX 3090 (24 GB) / NVIDIA L40

### Installation

```bash
# Clone the repository
git clone https://github.com/JSONWSZ/SFGPN.git
cd SFGPN

# Install dependencies
pip install -r SFGPN_noise/requirements.txt
```

**Dependency versions** (from [`SFGPN_noise/requirements.txt`](SFGPN_noise/requirements.txt)):

| Package | Version |
|---|---|
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| numpy | 2.4.1 |
| opencv-python | 4.11.0.86 |
| Pillow | 12.1.0 |
| scikit-learn | 1.8.0 |
| tensorboardX | 2.6.4 |
| thop | 0.1.1 |
| tqdm | 4.67.1 |
| matplotlib | 3.10.8 |
| grad-cam | 1.5.5 |
| ultralytics | 8.3.148 |

> **Note on `thop`**: Required only for FLOPs/Params measurement (`FLOPs_Params.py`). Not needed for training or testing.
> **Note on `ultralytics`**: Required only for mask generation (`PedestrianSegmentation/Segmentation.py`). Not needed if using our pre-generated masks.

---

## 4. Dataset Preparation

### 4.1 Original Datasets

| Dataset | Identities | Training Images | Source |
|---|---|---|---|
| **SYSU-MM01** [1] | 491 (395 train / 96 test) | 19,659 VIS + 12,792 NIR | [Download](http://isee.sysu.edu.cn/project/RGBIRReID.htm) |
| **RegDB** [2] | 412 | 4,120 VIS + 4,120 NIR | [Download](http://dm.dongguk.edu/link.html) |
| **LLCM** [3] | 1,064 (713 train / 351 test) | 16,946 VIS + 13,975 NIR | [Download](https://github.com/ZYK100/LLCM) (signed agreement required) |

### 4.2 Data Split Protocol

**SYSU-MM01**: Uses the official train/val/test split from `exp/train_id.txt`, `exp/val_id.txt`, and `exp/test_id.txt`. The training set combines the official train and val splits (395 identities total). Testing uses all-search and indoor-search modes following the standard protocol [1].

**RegDB**: Follows the standard 10-trial random split protocol. For each trial, the dataset is randomly split into 206 training and 206 testing identities. Results are averaged over 10 trials. Split files are located at `RegDB/idx/train_visible_{trial}.txt`, `RegDB/idx/train_thermal_{trial}.txt`, `RegDB/idx/test_visible_{trial}.txt`, `RegDB/idx/test_thermal_{trial}.txt`.

**LLCM**: Uses the official train/test split from `LLCM/idx/train_vis.txt`, `LLCM/idx/train_nir.txt`, and `LLCM/idx/test_id.txt`. Evaluated in both VIS→IR and IR→VIS modes.

### 4.3 Foreground Mask Generation

We use **YOLO11x-seg** instance segmentation model (from [Ultralytics](https://docs.ultralytics.com/tasks/segment/)) to generate body masks for all training images. The model detects pedestrians and personal belongings (backpacks, handbags, umbrellas, ties, suitcases — COCO classes 0, 24, 25, 26, 27, 28) and preserves them as foreground.

- **Script**: [`SFGPN_noise/PedestrianSegmentation/Segmentation.py`](SFGPN_noise/PedestrianSegmentation/Segmentation.py)
- **Model**: `yolo11x-seg.pt` (download automatically on first run)

**Dirty data filtering**: Original datasets contain "dirty" images (background only, no pedestrians). For SYSU-MM01, we set a confidence threshold of 0.1. Images below this threshold are saved separately (e.g., `cam1`, `cam2`, `cam5/0461` often contain empty backgrounds) and filtered during preprocessing.

### 4.4 Quick Start with Pre-processed Data

If you prefer not to run mask generation yourself, download our pre-processed datasets:

| Resource | Link |
|---|---|
| Processed data (instance segmented) | [Baidu Netdisk](https://pan.baidu.com/s/1Pw22313pqSaBGebggMH-uQ?pwd=1234) |

### 4.5 Preprocessing (SYSU-MM01)

After downloading the dataset and masks, run:

```bash
cd SFGPN_noise
python pre_process_sysu_mask.py
```

This generates the `SYSU-MM01-npy/` folder containing:
- `train_rgb_resized_img_pure.npy` — VIS training images
- `train_rgb_resized_img_mask.npy` — VIS foreground masks
- `train_rgb_resized_label.npy` — VIS labels
- `train_ir_resized_img_pure.npy` — IR training images
- `train_ir_resized_img_mask.npy` — IR foreground masks
- `train_ir_resized_label.npy` — IR labels

For RegDB and LLCM, images are loaded and resized on-the-fly in `data_loader.py`.

---

## 5. Reproducing Paper Results (Tables 1–9)

All experiments use **random seed = 0**. The complete configuration for reproducing each reported result is provided below.

### Common Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Input size | 384 × 144 |
| Batch size | 6 identities × 4 images/modality = 48 |
| Optimizer | SGD, momentum=0.9, weight_decay=5e-4, nesterov=True |
| GM Pool | off |
| Loss weights (fixed) | λ_tri=1.0, λ_id=1.0 |
| Random Erasing probability | 0.5 |
| BGS probability | 0.8 |
| BRE probability | 0.8 |

### Learning Rate Schedule

| Epoch Range | Learning Rate | Notes |
|---|---|---|
| 0–10 | 0.01 → 0.1 (warmup) | Basic losses only |
| 10–30 | 0.1 | Basic losses + SCG |
| 30–65 | 0.01 | Basic losses + SCG |
| 65–75 | 0.1 (reset) | WHC & WC activated at epoch 65 |
| 75–115 | 0.01 | All losses |
| 115–150 | 0.001 | All losses |

---

### Table 1 & 2: State-of-the-Art Comparison

**Table 1** (SYSU-MM01 and RegDB) and **Table 2** (LLCM) — the main benchmark results.

#### SYSU-MM01 (Full SFGPN)

```bash
# Training
python train.py --dataset sysu --gpu 0 --workers 4 --seed 0 --lr 0.1

# Testing (All Search)
python test_matrix.py --dataset sysu --mode all \
    --resume sysu_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4

# Testing (Indoor Search)
python test_matrix.py --dataset sysu --mode indoor \
    --resume sysu_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4
```

**Configuration**: FRL after layer2, CSFP after layer3, λ₁=0.4, λ₂=1.7, λ₃=0.24, seed=0, epochs=150.

**Expected results**:
| Setting | Rank-1 | Rank-10 | Rank-20 | mAP |
|---|---|---|---|---|
| All Search | 75.33% | 97.17% | 99.16% | 72.26% |
| Indoor Search | 83.80% | 98.84% | 99.88% | 85.87% |

#### RegDB

```bash
# Training (repeat for trial = 1, 2, ..., 10)
python train.py --dataset regdb --gpu 0 --workers 4 --seed 0 --trial 1

# Testing (VIS→IR mode), modify test_mode = [2, 1] in test_matrix.py
python test_matrix.py --dataset regdb --gpu 0

# Testing (IR→VIS mode), modify test_mode = [1, 2] in test_matrix.py
python test_matrix.py --dataset regdb --gpu 0
```

**Configuration**: FRL after layer2, CSFP before layer3 (RegDB-specific, pool_dim=1024), seed=0, epochs=150. Results averaged over 10 trials.

**Expected results** (average over 10 trials):
| Mode | Rank-1 | Rank-10 | Rank-20 | mAP |
|---|---|---|---|---|
| VIS→IR | 88.39% | 97.01% | 98.69% | 83.02% |
| IR→VIS | 86.67% | 95.76% | 97.75% | 81.78% |

> **Note**: RegDB uses a different CSFP insertion position (before layer3, with pool_dim=1024 and no layer4). This is handled automatically by setting `--dataset regdb`.

#### LLCM

```bash
# Training
python train.py --dataset llcm --gpu 0 --workers 4 --seed 0 --lr 0.1

# Testing (VIS→IR mode), set test_mode = [2, 1] in test_matrix.py
python test_matrix.py --dataset llcm \
    --resume llcm_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4

# Testing (IR→VIS mode), set test_mode = [1, 2] in test_matrix.py
python test_matrix.py --dataset llcm \
    --resume llcm_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4
```

**Expected results**:
| Mode | Rank-1 | Rank-10 | Rank-20 | mAP |
|---|---|---|---|---|
| IR→VIS | 55.60% | 85.06% | 91.40% | 62.47% |
| VIS→IR | 62.74% | 90.39% | 94.82% | 65.62% |

---

### Table 3: Ablation Study

Each row in Table 3 corresponds to a specific configuration. To reproduce each row:

#### Row 1: Baseline (ResNet50 only)
```
BGS=off, BRE=off, FRL+SCG=off, CSFP=off, WHC=off, WC=off
```
Train for 100 epochs (no center losses needed).
**Expected**: Rank-1=65.50%, mAP=62.34%

#### Row 2: Baseline + BGS
```
BGS=on, BRE=off, FRL+SCG=off, CSFP=off, WHC=off, WC=off
```
Modify `data_loader.py`: set `where="body"` for grayscale and disable erasing.

#### Row 3: Baseline + BRE
```
BGS=off, BRE=on, FRL+SCG=off, CSFP=off, WHC=off, WC=off
```

#### Row 4: Baseline + BGS + BRE
```
BGS=on, BRE=on, FRL+SCG=off, CSFP=off, WHC=off, WC=off
```
**Expected**: Rank-1=69.04%, mAP=66.05%

#### Row 5: + FRL + SCG (λ₁=0.08)
This is the configuration **before CSFP is added** (single branch, λ₁=0.08).
```
BGS=on, BRE=on, FRL+SCG=on, CSFP=off, WHC=off, WC=off
```
Train for 100 epochs with λ₁=0.08.
**Expected**: Rank-1=70.20%, mAP=67.23%

#### Row 6: + CSFP only
```
BGS=on, BRE=on, FRL+SCG=off, CSFP=on, WHC=off, WC=off
```
Train for 100 epochs. CSFP after layer3.
**Expected**: Rank-1=71.06%, mAP=68.99%

#### Row 7: + FRL + SCG (λ₁=0.4) + CSFP
```
BGS=on, BRE=on, FRL+SCG=on, CSFP=on, WHC=off, WC=off
```
Train for 100 epochs with λ₁=0.4 (re-tuned for 3-branch architecture).
**Expected**: Rank-1=72.85%, mAP=70.17%

#### Row 8: + WHC
```
BGS=on, BRE=on, FRL+SCG=on, CSFP=on, WHC=on, WC=off
```
Train for 150 epochs. λ₁=0.4, λ₂=1.7, WHC activated at epoch 65.
**Expected**: Rank-1=74.90%, mAP=71.84%

#### Row 9: Full SFGPN (+ WHC + WC)
```
BGS=on, BRE=on, FRL+SCG=on, CSFP=on, WHC=on, WC=on
```
Training command (default `train.py` with all components):
```bash
python train.py --dataset sysu --gpu 0 --workers 4 --seed 0
```
λ₁=0.4, λ₂=1.7, λ₃=0.24, WHC/WC activated at epoch 65.
**Expected**: Rank-1=75.33%, mAP=72.26%

---

### Tables 4 & 5: FRL/CSFP Insertion Position

**Table 4 — FRL position**: Fix λ₁=1.0, train 100 epochs, vary FRL insertion layer in `model.py`.

| Position | Code change in `model.py` |
|---|---|
| After layer1 | Insert FRL between layer1 and layer2 |
| After layer2 | Default (used in final model) |
| After layer3 | Insert FRL between layer3 and layer4 |
| After layer4 | Insert FRL after layer4 |

**Expected**:
| Position | Rank-1 | mAP |
|---|---|---|
| After layer1 | 64.92% | 62.21% |
| After layer2 | 69.47% | 66.32% |
| After layer3 | 68.34% | 66.01% |
| After layer4 | 66.18% | 65.86% |

**Table 5 — CSFP position**: With FRL fixed at layer2 and λ₁=0.08, vary CSFP insertion layer.

| Position | Code change in `model.py` |
|---|---|
| After layer2 | Place `self.multiscale` call after layer2 |
| After layer3 | Default for SYSU/LLCM |
| After layer4 | Place `self.multiscale` call after layer4 |

**Expected**:
| Position | Rank-1 | mAP |
|---|---|---|
| After layer2 | 67.41% | 64.59% |
| After layer3 | 71.41% | 69.27% |
| After layer4 | 66.02% | 64.51% |

---

### Table 6: CSFP Branch Loss Strategy

Compare concatenated vs. separated loss computation. This is controlled in `train.py` lines 288–292 (separated) vs. a modified version that computes losses on the concatenated batch.

- **Concatenated**: Compute triplet and center losses jointly on `feat` (3B × dim)
- **Separated (ours)**: Split `feat` into 3 chunks, compute losses independently, then average

**Expected**:
| Strategy | Rank-1 | mAP |
|---|---|---|
| Concatenated | 71.01% | 68.36% |
| Separated (ours) | 71.41% | 69.27% |

---

### Table 7 & Fig. 6: Hyperparameter Analysis

The three hyperparameters are tuned sequentially:

**λ₁ (SCG weight)**: With FRL+CSFP (no WHC/WC), sweep λ₁ ∈ {0.04, 0.08, 0.2, 0.4, 0.6, 0.8, 1.0}. Modify `train.py` line 313: `0.4 * loss_F` → sweep value. Best: **0.4**.

**λ₂ (WHC weight)**: With FRL+CSFP+SCG (λ₁=0.4), sweep λ₂ ∈ {0.5, 1.0, 1.5, 1.7, 2.0}. Modify `train.py` line 311: `1.7 * loss_HC` → sweep value. Best: **1.7**.

**λ₃ (WC weight)**: With FRL+CSFP+SCG+WHC (λ₁=0.4, λ₂=1.7), sweep λ₃ ∈ {0.08, 0.16, 0.24, 0.32, 0.4}. Modify `train.py` line 311: `0.24 * loss_twCompact` → sweep value. Best: **0.24**.

---

### Table 8: Loss Introduction Schedule

Compare WHC/WC activation at epoch 1 vs. epoch 65.

- **Epoch 1**: Modify `train.py` to activate WHC/WC from the start (change `if epoch >= 65` to `if epoch >= 1`)
- **Epoch 65 (ours)**: Default behavior

**Expected**:
| Introduction Time | Rank-1 | mAP |
|---|---|---|
| Epoch 1 | 71.26% | 69.38% |
| Epoch 65 (ours) | 72.87% | 70.57% |

> **Note**: For this ablation, WHC and WC weights are fixed to 0.01 (small value to isolate the schedule effect).

---

### Table 9: Weighted Identity Center Ablation

Compare mean-based vs. cosine-similarity-weighted identity centers in `loss.py`.

- **Averaged**: Replace `weighted_center()` in `modality_center_loss()` with a simple `feats_grp.mean(dim=0)`
- **Weighted (ours)**: Default `weighted_center()` using cosine similarity weights

**Expected**:
| Center Type | Rank-1 | mAP |
|---|---|---|
| Averaged | 72.49% | 70.09% |
| Weighted (ours) | 72.87% | 70.57% |

---

### Table 10: SCG Loss Decomposition

The SCG loss consists of an L2 (BCE) term and a Dice term. To isolate each:

- **L2 only**: Comment out the Dice term in `FocalLoss.forward()` (line 166–171 in `loss.py`)
- **Dice only**: Comment out the BCE term (line 162 in `loss.py`)
- **L2 + Dice (full)**: Default

**Expected**:
| Configuration | Rank-1 | mAP |
|---|---|---|
| L2 only | 73.70% | 70.21% |
| Dice only | 74.49% | 70.87% |
| L2 + Dice (full) | 75.33% | 72.26% |

---

### Table 11: FRL Heatmap Quantitative Evaluation

Uses [`eval_frl_heatmap.py`](eval_frl_heatmap.py) to compute mIoU, Dice, Foreground Recall, BSR, and ACR between FRL heatmaps and YOLO11 masks.

```bash
python eval_frl_heatmap.py \
    --dataset_path ../Datasets/SYSU-MM01 \
    --pedestrian_path ../Datasets/SYSU-MM01-Pedestrian \
    --heatmap_save_path ../Datasets/SYSU-MM01-heatmap \
    --checkpoint SFGPN_noise/save_model/sysu_agw_p4_n6_lr_0.1_seed_0_best.t \
    --gpu 0
```

**Expected results** (averaged over SYSU-MM01 test set):
| Metric | Value |
|---|---|
| mIoU | 0.777 |
| Dice score | 0.869 |
| Foreground Recall | 0.881 |
| Background Suppression Ratio (BSR) | 20.93 |
| Attention Concentration Ratio (ACR) | 1.63 |

---

### Table 12: Mask Noise Robustness

Use the `--noise_kernel` argument in `train.py` to apply random erosion/dilation to body masks:

```bash
# Clean (k=0)
python train.py --dataset sysu --gpu 0 --seed 0 --noise_kernel 0

# Light noise (k=5)
python train.py --dataset sysu --gpu 0 --seed 0 --noise_kernel 5

# Medium noise (k=11)
python train.py --dataset sysu --gpu 0 --seed 0 --noise_kernel 11

# Heavy noise (k=21)
python train.py --dataset sysu --gpu 0 --seed 0 --noise_kernel 21
```

Mask noise is implemented in [`SFGPN_noise/data_loader.py`](SFGPN_noise/data_loader.py) function `add_mask_noise()` (line 12).

**Expected**:
| Noise | SYSU-MM01 R-1 | SYSU-MM01 mAP | RegDB R-1 | RegDB mAP |
|---|---|---|---|---|
| None (k=0) | 75.33% | 72.26% | 89.54% | 80.89% |
| k=5 | 74.71% | 71.45% | 89.78% | 81.15% |
| k=11 | 73.88% | 70.61% | 89.87% | 81.27% |
| k=21 | 74.03% | 70.74% | 89.57% | 81.00% |

---

### Table 13: Reusability (Plug-and-Play)

The FRL module and SCG loss are integrated into existing methods (AGW, MCLNet, MMN) located in their respective subdirectories:

| Baseline | Directory | SCG Weight | Expected R-1 Gain |
|---|---|---|---|
| AGW | `AGW/` | 0.04 | 47.50% → 51.55% (+4.05) |
| MCLNet | `MCLNet/` | 0.05 | 65.40% → 68.75% (+3.35) |
| MMN | `MMN/` | 0.008 | 70.60% → 71.06% (+0.46) |

Each baseline directory contains the FRL-augmented training script. FRL is inserted after the second backbone stage of each method, and SCG weight is individually tuned on SYSU-MM01.

---

### Table 14: Model Complexity Analysis

Run the provided analysis scripts:

```bash
# FLOPs and Parameters
python SFGPN_noise/FLOPs_Params.py

# Inference Time and FPS
python SFGPN_noise/Inference_Time.py
```

**Benchmarking protocol**: All methods are benchmarked on the same workstation (NVIDIA L40 GPU, Intel Xeon Gold 6148 CPU @ 2.40GHz) with identical input size 384×144. FLOPs/Params are computed using `thop.profile` on a random tensor of shape (1, 3, 384, 144). Inference time is measured as the average over 10,000 forward passes after 1,000 warm-up runs.

**Expected**:
| Method | FLOPs (G) | Params (M) | Inference (ms) | FPS |
|---|---|---|---|---|
| AGW | 6.90 | 23.54 | 6.80 | 146.99 |
| SFGPN (ours) | 15.02 | 32.16 | 5.42 | 184.36 |

> **Note**: FRL is discarded at inference. The inference time records the duration from data input to embedding output (feature extraction only, not including I/O or retrieval search).

---

## 6. Training

### SYSU-MM01

```bash
cd SFGPN_noise
python pre_process_sysu_mask.py    # First time only
python train.py --dataset sysu --gpu 0 --workers 4 --seed 0
```

### LLCM

```bash
python train.py --dataset llcm --gpu 0 --workers 4 --seed 0 --lr 0.1
```

### RegDB

```bash
# Run for each trial (1–10)
python train.py --dataset regdb --gpu 0 --workers 4 --seed 0 --trial 1
# ... trial 2 through 10
```

### Training Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset: `sysu`, `regdb`, or `llcm` | `sysu` |
| `--lr` | Initial learning rate | `0.1` |
| `--arch` | Backbone architecture | `resnet50` |
| `--batch-size` | Number of identities per batch | `6` |
| `--num_pos` | Images per identity per modality | `4` |
| `--gpu` | GPU device ID | `0` |
| `--workers` | Data loading workers | `4` |
| `--seed` | Random seed | `0` |
| `--trial` | Trial number (RegDB only) | `1` |
| `--noise_kernel` | Mask noise kernel size (0=clean) | `0` |
| `--resume` | Resume from checkpoint | `''` |

### Output Files

Training produces:
- **Model checkpoints**: `save_model/{dataset}_agw_p4_n6_lr_0.1_seed_0[_trial_{trial}]_best.t` — best model by Rank-1
- **Final epoch model**: `save_model/{dataset}_agw_p4_n6_lr_0.1_seed_0_epoch_150.t`
- **Training logs**: `log/{dataset}_log/{suffix}_os.txt` — full console output
- **TensorBoard logs**: `log/vis_log/{suffix}/` — loss curves, accuracy curves

---

## 7. Testing

### SYSU-MM01

```bash
# All Search mode
python test_matrix.py --dataset sysu --mode all \
    --resume sysu_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4

# Indoor Search mode
python test_matrix.py --dataset sysu --mode indoor \
    --resume sysu_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4
```

### LLCM

```bash
# VIS→IR mode: before running, set test_mode = [2, 1] in test_matrix.py
python test_matrix.py --dataset llcm \
    --resume llcm_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4

# IR→VIS mode: before running, set test_mode = [1, 2] in test_matrix.py
python test_matrix.py --dataset llcm \
    --resume llcm_agw_p4_n6_lr_0.1_seed_0_best.t --gpu 0 --workers 4
```

### RegDB

For RegDB, modify `test_matrix.py` to set the test direction:

**VIS→IR mode**:
```python
test_mode = [2, 1]
query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
```

**IR→VIS mode**:
```python
test_mode = [1, 2]
query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
```

```bash
python test_matrix.py --dataset regdb --gpu 0
```

### Testing Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset: `sysu`, `regdb`, or `llcm` | `sysu` |
| `--mode` | Search mode for SYSU: `all` or `indoor` | `all` |
| `--resume` | Checkpoint filename | `sysu_agw_p4_n6_lr_0.1_seed_0_best.t` |
| `--gpu` | GPU device ID | `0` |
| `--workers` | Data loading workers | `4` |
| `--test-batch` | Testing batch size | `32` |

### Evaluation Metrics

All evaluation uses three complementary metrics following the standard VI-ReID protocol:
- **CMC (Rank-k accuracy)**: Cumulative Matching Characteristics at ranks 1, 5, 10, 20
- **mAP**: mean Average Precision
- **mINP**: mean Inverse Negative Penalty

For SYSU-MM01 and LLCM, we follow the single-shot all-search setting. Gallery images sharing both the same identity and camera as the query are excluded from evaluation. Results for SYSU-MM01 and LLCM are averaged over 10 random gallery samplings. For RegDB, results are averaged over 10 train/test splits.

---

## 8. Visualization & Analysis Scripts

| Script | Purpose |
|---|---|
| `Grad-CAM.py` | Multi-branch Grad-CAM attention visualization |
| `Grad-CAM-plusplus.py` | Grad-CAM++ visualization |
| `t-SNE.py` | t-SNE feature distribution visualization |
| `plot_intra_inter.py` | Intra-class vs. inter-class distance histograms |
| `test_query_SYSU.py` | Top-10 retrieval results for SYSU-MM01 (with saved images) |
| `test_query_RegDB.py` | Top-10 retrieval results for RegDB (with saved images) |
| `test_heatmap_RE.py` | FRL heatmap visualization under random erasing |
| `test_mask.py` | Mask and heatmap overlay visualization |
| `bar.py` | Bar chart for metric comparison |
| `eval_frl_heatmap.py` | Quantitative FRL heatmap evaluation (Table 11) |
| `FLOPs_Params.py` | FLOPs and parameter count measurement (Table 14) |
| `Inference_Time.py` | Inference time and FPS measurement (Table 14) |

---

## 9. Pretrained Models & Processed Data

| Resource | Google Drive | Baidu Netdisk |
|---|---|---|
| Trained models (all datasets) | [GoogleDrive](https://drive.google.com/drive/folders/1guJetD4OFoohCVma5rUBq2TK9Yxhuv1F?usp=drive_link) | [Baidu Netdisk](https://pan.baidu.com/s/1VCFskf1Kx0rANla0l9tD1g?pwd=1234) |
| Processed data (instance segmented) | — | [Baidu Netdisk](https://pan.baidu.com/s/1Pw22313pqSaBGebggMH-uQ?pwd=1234) |

### Checkpoint Naming Convention

```
{dataset}_agw_p4_n6_lr_0.1_seed_0[_trial_{trial}][_noise_k{size}]_best.t
```

Examples:
- `sysu_agw_p4_n6_lr_0.1_seed_0_best.t` — SYSU-MM01, clean masks
- `regdb_agw_p4_n6_lr_0.1_seed_0_trial_1_best.t` — RegDB, trial 1
- `llcm_agw_p4_n6_lr_0.1_seed_0_best.t` — LLCM, clean masks
- `sysu_agw_p4_n6_lr_0.1_seed_0_noise_k5_best.t` — SYSU-MM01, mask noise k=5

---

## 10. Citation

If this work and code are helpful for your research, please cite our paper:

```bibtex
@article{Wu2026SFGPN,
  title={Soft Foreground-Guided Cross-Scale Perception for Visible–Infrared Person Re-Identification},
  author={Wu, Shuzuan and Zhang, Chengfang and Feng, Ziliang},
  journal={The Visual Computer},
  year={2026},
  doi={10.5281/zenodo.18376063}
}
```

The code and data are permanently archived at Zenodo: [10.5281/zenodo.18376063](https://doi.org/10.5281/zenodo.18376063).

---

## 11. Acknowledgments

This work is supported by the Sichuan Science and Technology Program (2024NSFSC2046), the Intelligent Policing Key Laboratory of Sichuan Province (ZNJW2025KFZD001), and the Luzhou Science and Technology Program (2025JYJ049).

Most of the code is based on [DEEN](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3]. We thank the authors for their contributions to the community.

### References

[1] A. Wu et al. RGB-infrared cross-modality person re-identification. *ICCV*, 2017.

[2] D. T. Nguyen et al. Person recognition system based on a combination of body images from visible light and thermal cameras. *Sensors*, 17(3):605, 2017.

[3] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification. *CVPR*, 2023.
