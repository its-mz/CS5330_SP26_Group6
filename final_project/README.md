# Unsupervised Learning for Medical Image Registration

### Reproducing VoxelMorph (Balakrishnan et al., IEEE TMI 2019)

**CS 5330 Pattern Recognition and Computer Vision**  
**Group 6 — Jia Song, Yi-hsuan Lai, Mingzhe Ou**  
**Northeastern University, Spring 2026**

---

## Project Overview

This project reproduces **VoxelMorph**, a deep learning framework for unsupervised deformable brain MRI registration. Given two 3D T1-weighted brain MRI volumes (a _fixed_ image and a _moving_ image), the model predicts a dense displacement vector field (DVF) that spatially aligns them, enabling accurate cross-subject brain alignment without requiring ground-truth deformation labels.

### Key Results

| Metric                  | Before Registration | After Registration | Improvement |
| ----------------------- | ------------------- | ------------------ | ----------- |
| Mean DSC (3 structures) | 0.547               | **0.601**          | **+0.053**  |
| CSF DSC                 | 0.436               | 0.489              | +0.053      |
| Gray Matter DSC         | 0.595               | 0.631              | +0.036      |
| White Matter DSC        | 0.610               | 0.682              | +0.072      |
| Test pairs improved     | —                   | **20 / 20**        | 100%        |

> Model trained on 39 OASIS-1 subjects (128³ resolution, 150 epochs, T4 GPU)

---

## Project Type

**Reproducing Project** — We reproduce the VoxelMorph framework using the official PyTorch implementation, train on the OASIS-1 dataset, and evaluate using Dice Similarity Coefficient (DSC) on FSL brain segmentation labels.

> Note: The original paper used TensorFlow. Due to tensorflow-macos incompatibility with Apple Silicon, we use the official PyTorch version (`voxelmorph/voxelmorph` main branch), which has an architecturally identical U-Net + STN implementation.

---

## Repository Structure

```
final_project/
├── src/
│   ├── preprocess.py         # NIfTI loading and intensity normalization
│   ├── load_model.py         # VoxelMorph model initialization
│   ├── train.py              # Full training pipeline (NCC + gradient loss)
│   ├── inference.py          # Inference with random-initialized model (Phase 1)
│   ├── inference_trained.py  # Inference with trained weights (Phase 2)
│   ├── evaluate.py           # DSC evaluation + Jacobian determinant analysis
│   └── visualize.py          # Before/after registration visualization
├── data/
│   ├── train/                # Preprocessed OASIS-1 .nii.gz volumes
│   └── test/                 # Test volumes (fixed_001, moving_001)
├── weights/
│   └── voxelmorph_final.pt   # Trained model weights (see Dataset section)
├── outputs/
│   ├── registered/           # Phase 1 inference outputs
│   ├── registered_trained/   # Phase 2 inference outputs
│   └── eval/                 # DSC plots, training loss curves
└── README.md
```

---

## Environment Setup

### Requirements

- macOS (Apple Silicon M1/M2/M3) or Linux
- Python 3.10
- Anaconda / Miniconda

### Installation

```bash
# 1. Create conda environment
conda create -n voxelmorph python=3.10 -y
conda activate voxelmorph

# 2. Install TensorFlow for Apple Silicon (if needed for other tasks)
# pip install tensorflow-macos==2.12.0 tensorflow-metal==0.8.0

# 3. Install core dependencies
pip install "numpy==1.23.5"
pip install "nibabel==4.0.2"
pip install SimpleITK scipy matplotlib pandas tqdm

# 4. Install VoxelMorph (PyTorch version)
pip install git+https://github.com/adalca/neurite.git --no-deps
pip install git+https://github.com/voxelmorph/voxelmorph.git --no-deps
pip install einops pystrum torch --no-deps

# 5. Verify installation
python -c "import voxelmorph as vxm; import nibabel;"
```

### Known Issues

- `tensorflow-macos 2.12` requires `numpy < 1.24`, which conflicts with newer packages. Always pin `numpy==1.23.5` after installation.
- VoxelMorph's `dev-tensorflow` branch requires a newer version of `neurite` than what's on PyPI. Use the PyTorch version (main branch) instead.
- GPU training requires Google Colab or an NVIDIA GPU. Apple Silicon MPS is not supported by VoxelMorph's 3D operations.

---

## Dataset

### OASIS-1 (Open Access Series of Imaging Studies)

- **Subjects:** 416 T1-weighted brain MRI scans, ages 18–96
- **Access:** Requires free registration at [oasis-brains.org](https://www.oasis-brains.org/#access)
- **Download:** After approval, download from [OASIS-1 page](https://sites.wustl.edu/oasisbrains/home/oasis-1/)
  - Raw Data: `oasis_cross-sectional_disc1.tar.gz` (~1.5 GB)
- **Preprocessed data (Google Drive):** [Link to preprocessed NIfTI files](https://drive.google.com/drive/folders/1_PC2FVX6CnxBFkmYKaaRQfVRzuxrQBC7?usp=drive_link)
  _(39 subjects preprocessed to 160×192×224, normalized to [0,1])_

### Trained Model Weights

Download `voxelmorph_final.pt` from Google Drive:  
**[weights/voxelmorph_final.pt](https://drive.google.com/drive/folders/140PHrx1keRFv7Dz8GM108d38dePSnZxV?usp=drive_link)**

Place in the `weights/` directory before running inference.

### Data Structure (after download)

```
data/
├── raw/
│   └── disc1/
│       ├── OAS1_0001_MR1/
│       │   ├── PROCESSED/MPRAGE/T88_111/*_masked_gfc.img  ← MRI (skull stripped)
│       │   └── FSL_SEG/*_fseg.img                         ← Segmentation labels
│       └── OAS1_0002_MR1/ ...
└── train/
    ├── OAS1_0001_MR1.nii.gz   ← preprocessed (160×192×224, [0,1])
    └── OAS1_0002_MR1.nii.gz ...
```

### Preprocessing Steps Applied

1. Load Analyze format (`.hdr` + `.img`) using nibabel
2. Select skull-stripped, MNI-space aligned volume (`*_masked_gfc.img`)
3. Squeeze extra dimensions if 4D
4. Center crop/pad to 160×192×224
5. Min-max intensity normalization to [0, 1]
6. Save as compressed NIfTI (`.nii.gz`)

---

## Running the Code

### Step 1: Preprocess OASIS data (if starting from raw)

```bash
cd final_project

# After downloading and extracting disc1.tar.gz to data/raw/
python -c "
import os, glob, numpy as np, nibabel as nib

def crop_or_pad(vol, target=(160,192,224)):
    for i in range(3):
        s, t = vol.shape[i], target[i]
        # ... (see src/preprocess.py for full implementation)
    return vol

# Run preprocessing
# See src/preprocess.py for full script
"
```

### Step 2: Train VoxelMorph (requires GPU — use Google Colab)

```bash
# Local (CPU only — very slow, not recommended for full training)
python src/train.py \
    --data_dir data/train \
    --out_dir  weights \
    --epochs   150 \
    --size     128 \
    --lr       1e-4 \
    --lam      0.5 \
    --save_every 50

# Recommended: Run on Google Colab with T4 GPU
# Upload src/train.py and data/train/ to Google Drive
# Expected training time: ~1-2 hours on T4 GPU
```

### Step 3: Run Inference

```bash
# Phase 1 demo (random initialized model)
python src/inference.py

# Phase 2 (trained model — requires weights/voxelmorph_final.pt)
python src/inference_trained.py
```

### Step 4: Evaluate DSC

```bash
# Requires raw OASIS data with FSL_SEG labels
python src/evaluate.py \
    --data_dir data/raw/disc1 \
    --weights  weights/voxelmorph_final.pt \
    --n_pairs  20 \
    --size     128
```

### Step 5: Visualize Results

```bash
python src/visualize.py
```

---

## Module Descriptions

| Script                 | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| `preprocess.py`        | Load NIfTI/Analyze volumes, normalize intensity, crop/pad to target shape   |
| `load_model.py`        | Initialize VxmPairwise model with specified architecture                    |
| `train.py`             | Training loop with NCC similarity loss + gradient regularization            |
| `inference.py`         | Run inference with randomly initialized model (Phase 1 demo)                |
| `inference_trained.py` | Run inference with trained weights, resize output back to original shape    |
| `evaluate.py`          | Compute DSC on FSL labels (CSF/GM/WM), Jacobian determinant, generate plots |
| `visualize.py`         | 6-panel visualization: Fixed / Moving / Registered + difference maps        |

---

## Method Overview

### Pipeline

```
OASIS-1 NIfTI volumes
        ↓
  Preprocessing (normalize, resize to 128³)
        ↓
  Classical Baseline: ANTsPy affine (Mutual Information)
        ↓
  VoxelMorph U-Net CNN
  • Input: concat(moving, fixed) → shape (1,2,128,128,128)
  • Encoder: 16→32→32→32 features, stride-2 downsampling
  • Decoder: symmetric upsampling with skip connections
  • Output: velocity field → scaling & squaring → DVF (3,128,128,128)
        ↓
  Spatial Transformer Network (trilinear interpolation)
        ↓
  Registered image + DVF
        ↓
  Evaluation: DSC on FSL labels (CSF, GM, WM)
              Jacobian determinant (deformation regularity)
```

### Loss Function

```
L = L_sim + λ · L_reg

L_sim = -NCC(fixed, registered)   # Normalized Cross-Correlation
L_reg = ||∇DVF||                  # Gradient smoothness regularization
λ = 0.5                           # Regularization weight (matches paper)
```

---

## Results

### Quantitative (DSC — 20 test pairs)

```
Mean DSC before registration : 0.547
Mean DSC after  registration : 0.601  (+0.053)
Test pairs improved          : 20/20 (100%)

Per-structure:
  CSF          : 0.436 → 0.489  (+0.053)
  Gray Matter  : 0.595 → 0.631  (+0.036)
  White Matter : 0.610 → 0.682  (+0.072)
```

### Comparison with Paper

|                    | Original Paper           | Our Reproduction  |
| ------------------ | ------------------------ | ----------------- |
| Mean DSC (after)   | ~0.750                   | 0.601             |
| Training subjects  | 394                      | 39 (10% of paper) |
| Label granularity  | 35 FreeSurfer structures | 3 FSL structures  |
| Resolution         | 160×192×224              | 128³              |
| All pairs improved | ✅                       | ✅                |

The gap is expected given 10× less training data and coarser labels. The trend — consistent DSC improvement across all test pairs — matches the paper's core finding.

---

## References

1. Balakrishnan, G., et al. "VoxelMorph: A Learning Framework for Deformable Medical Image Registration." _IEEE TMI_, 2019.
2. Marcus, D.S., et al. "Open Access Series of Imaging Studies (OASIS)." _Journal of Cognitive Neuroscience_, 2007.
3. Avants, B.B., et al. "Symmetric Diffeomorphic Image Registration with Cross-Correlation." _Medical Image Analysis_, 2008.
