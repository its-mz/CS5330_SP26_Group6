import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

def visualize(fixed_path, moving_path, registered_path, out_path):
    fixed = nib.load(fixed_path).get_fdata()
    moving = nib.load(moving_path).get_fdata()
    registered = nib.load(registered_path).get_fdata()

    sl = fixed.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: three volumes
    for ax, img, title in zip(axes[0],
        [fixed[:,:,sl], moving[:,:,sl], registered[:,:,sl]],
        ['Fixed', 'Moving (before)', 'Registered (after)']):
        ax.imshow(img.T, cmap='gray', origin='lower')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Row 2: difference maps
    diff_before = np.abs(fixed[:,:,sl] - moving[:,:,sl])
    diff_after  = np.abs(fixed[:,:,sl] - registered[:,:,sl])
    improvement = diff_before - diff_after

    for ax, img, title, cmap in zip(axes[1],
        [diff_before, diff_after, improvement],
        ['Fixed - Moving', 'Fixed - Registered', 'Improvement (green=better)'],
        ['hot', 'hot', 'RdYlGn']):
        ax.imshow(img.T, cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.suptitle('VoxelMorph Registration — Phase 1 Demo', fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    visualize(
        fixed_path = "data/test/fixed_001.nii.gz",
        moving_path = "data/test/moving_001.nii.gz",
        registered_path = "outputs/registered/registered.nii.gz",
        out_path = "outputs/eval/registration_viz.png"
    )
