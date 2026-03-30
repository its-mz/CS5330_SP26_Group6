import torch
import numpy as np
import nibabel as nib
import voxelmorph as vxm
import torch.nn.functional as F
import os

def load_trained_model(weights_path, device='cpu'):
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 32, 32, 32],
        integration_steps=7,
        device=device
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded weights from: {weights_path}")
    return model

def preprocess(path, target_size=128):
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    t = torch.tensor(vol).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_size,)*3, mode='trilinear', align_corners=True)
    return t.squeeze(0)  # (1,128,128,128)

def save_nifti(arr, out_path):
    nib.save(nib.Nifti1Image(arr, np.eye(4)), out_path)

def run_inference(fixed_path, moving_path, weights_path, out_dir):
    print("Loading model...")
    model = load_trained_model(weights_path)

    print("Preprocessing volumes...")
    fixed  = preprocess(fixed_path).unsqueeze(0)   # (1,1,128,128,128)
    moving = preprocess(moving_path).unsqueeze(0)

    print("Running inference...")
    with torch.no_grad():
        dvf, registered = model(moving, fixed, return_warped_source=True)

    reg_np = registered[0, 0].numpy()
    dvf_np = dvf[0].permute(1,2,3,0).numpy()

    os.makedirs(out_dir, exist_ok=True)
    save_nifti(reg_np, f"{out_dir}/registered.nii.gz")
    save_nifti(dvf_np[...,0], f"{out_dir}/dvf_x.nii.gz")

    print(f"registered shape: {reg_np.shape}")
    print(f"registered range: {reg_np.min():.3f} ~ {reg_np.max():.3f}")
    print(f"Saved to {out_dir}/")
    print("Done ✓")

if __name__ == "__main__":
    run_inference(
        fixed_path = "data/test/fixed_001.nii.gz",
        moving_path = "data/test/moving_001.nii.gz",
        weights_path = "weights/voxelmorph_final.pt",
        out_dir = "outputs/registered_trained"
    )
