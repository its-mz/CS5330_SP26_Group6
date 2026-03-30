import os
import numpy as np
import nibabel as nib
import torch
from load_model import build_voxelmorph_model


def preprocess(path):
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return torch.tensor(vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)


def save_nifti(arr, ref_path, out_path):
    ref = nib.load(ref_path)
    nib.save(nib.Nifti1Image(arr, ref.affine), out_path)


def run_inference(fixed_path, moving_path, out_dir):
    print("Loading volumes...")
    fixed = preprocess(fixed_path)
    moving = preprocess(moving_path)
    print(f"  Volume shape: {tuple(fixed.shape[2:])}")

    print("Building model...")
    model = build_voxelmorph_model()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Running inference...")
    with torch.no_grad():
        dvf, registered = model(
            moving, fixed,
            return_warped_source=True
        )

    registered = registered[0, 0].numpy()   # (H,W,D)
    dvf_np     = dvf[0].permute(1,2,3,0).numpy()  # (H,W,D,3)

    os.makedirs(out_dir, exist_ok=True)
    save_nifti(registered,   fixed_path, f"{out_dir}/registered.nii.gz")
    save_nifti(dvf_np[...,0], fixed_path, f"{out_dir}/dvf_x.nii.gz")

    print(f"  registered shape : {registered.shape}")
    print(f"  registered range : {registered.min():.3f} ~ {registered.max():.3f}")
    print(f"  DVF shape        : {dvf_np.shape}")
    print(f"Saved to {out_dir}/")
    print("Done ✓")
    return registered, dvf_np


if __name__ == "__main__":
    run_inference(
        fixed_path = "data/test/fixed_001.nii.gz",
        moving_path = "data/test/moving_001.nii.gz",
        out_dir = "outputs/registered"
    )