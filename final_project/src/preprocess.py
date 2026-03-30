import numpy as np
import nibabel as nib


def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI volume and return as float32 numpy array."""
    vol = nib.load(path).get_fdata()
    return vol.astype(np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize volume to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    return (vol - vmin) / (vmax - vmin + 1e-8)


def add_channel_dim(vol: np.ndarray) -> np.ndarray:
    """Add trailing channel dim: (H,W,D) -> (H,W,D,1)"""
    return vol[..., np.newaxis]


def preprocess_volume(path: str) -> np.ndarray:
    """Full preprocessing pipeline for a single volume."""
    vol = load_volume(path)
    vol = normalize(vol)
    vol = add_channel_dim(vol)
    return vol   # shape: (160, 192, 224, 1)


def preprocess_segmentation(path: str) -> np.ndarray:
    """Load segmentation label map (no normalization, keep integer labels)."""
    seg = nib.load(path).get_fdata()
    return seg.astype(np.float32)   # shape: (160, 192, 224)


if __name__ == "__main__":
    # Quick sanity check
    import glob
    test_vols = sorted([f for f in
        glob.glob('../data/test/*.nii.gz') if '_seg' not in f])
    
    vol = preprocess_volume(test_vols[0])
    print(f"Preprocessed volume shape: {vol.shape}")   # (160,192,224,1)
    print(f"Intensity range: [{vol.min():.3f}, {vol.max():.3f}]")
    print("preprocess.py ✓")