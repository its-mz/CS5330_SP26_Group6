import os
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import voxelmorph as vxm

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_volume(path: str, size: int = 128) -> torch.Tensor:
    """
    Load and preprocess a NIfTI MRI volume.
    Normalizes intensity to [0,1] and resizes to size^3.
    Returns tensor of shape (1, 1, size, size, size).
    """
    arr = nib.load(path).get_fdata().astype(np.float32)
    while arr.ndim > 3:
        arr = arr[..., 0]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(size,) * 3, mode='trilinear', align_corners=True)


def load_segmentation(path: str, size: int = 128) -> torch.Tensor:
    """
    Load a FSL segmentation label map.
    Uses nearest-neighbor interpolation to preserve discrete labels.
    Labels: 0=background, 1=CSF, 2=Gray Matter, 3=White Matter
    Returns tensor of shape (1, 1, size, size, size).
    """
    arr = nib.load(path).get_fdata().astype(np.float32)
    while arr.ndim > 3:
        arr = arr[..., 0]
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(size,) * 3, mode='nearest')


# ── Warping ───────────────────────────────────────────────────────────────────

def warp_segmentation(seg: torch.Tensor, dvf: torch.Tensor) -> torch.Tensor:
    """
    Apply displacement vector field (DVF) to segmentation map
    using nearest-neighbor interpolation via scipy.map_coordinates.

    Args:
        seg: segmentation tensor (1, 1, H, W, D)
        dvf: displacement field tensor (1, 3, H, W, D)
    Returns:
        warped segmentation tensor (1, 1, H, W, D)
    """
    seg_np = seg.squeeze().cpu().numpy()      # (H, W, D)
    dvf_np = dvf[0].cpu().numpy()             # (3, H, W, D)

    H, W, D = seg_np.shape
    coords = np.mgrid[0:H, 0:W, 0:D].astype(np.float32)
    coords[0] += dvf_np[0]
    coords[1] += dvf_np[1]
    coords[2] += dvf_np[2]

    warped = map_coordinates(seg_np, coords, order=0, mode='nearest')
    return torch.tensor(warped).unsqueeze(0).unsqueeze(0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor, labels: list = [1, 2, 3]) -> tuple:
    """
    Compute Dice Similarity Coefficient for each label.

    DSC = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred:   predicted segmentation tensor
        target: ground truth segmentation tensor
        labels: list of label IDs to evaluate
    Returns:
        (per_label_scores, mean_score)
    """
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    scores = []
    for l in labels:
        p = (pred_np == l)
        t = (target_np == l)
        intersection = (p & t).sum()
        dsc = 2 * intersection / (p.sum() + t.sum() + 1e-8)
        scores.append(float(dsc))

    return scores, float(np.mean(scores))


def jacobian_determinant_stats(dvf: torch.Tensor) -> dict:
    """
    Compute Jacobian determinant statistics of the displacement field.
    Negative values indicate folding (anatomically implausible).

    Args:
        dvf: displacement field tensor (1, 3, H, W, D)
    Returns:
        dict with mean, std, and percentage of negative values
    """
    dvf_np = dvf[0].cpu().numpy()  # (3, H, W, D)

    dy = dvf_np[0, 1:, :, :] - dvf_np[0, :-1, :, :]
    dx = dvf_np[1, :, 1:, :] - dvf_np[1, :, :-1, :]
    dz = dvf_np[2, :, :, 1:] - dvf_np[2, :, :, :-1]

    s = min(dy.shape[0], dx.shape[0], dz.shape[0],
            dy.shape[1], dx.shape[1], dz.shape[1],
            dy.shape[2], dx.shape[2], dz.shape[2])

    jac = ((1 + dy[:s, :s, :s]) *
           (1 + dx[:s, :s, :s]) *
           (1 + dz[:s, :s, :s]))

    neg_pct = float((jac < 0).sum() / jac.size * 100)
    return {
        'mean': float(jac.mean()),
        'std':  float(jac.std()),
        'neg_pct': neg_pct
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Using device: {device}")

    # Load model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 32, 32, 32],
        integration_steps=7,
        device=str(device)
    ).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"[Model] Loaded weights: {args.weights}")

    # Find subject directories
    subjects = sorted(glob.glob(os.path.join(args.data_dir, 'OAS1_*')))
    print(f"[Data] Found {len(subjects)} subjects")

    label_names = ['CSF', 'Gray Matter', 'White Matter']
    results = []

    for i in range(min(len(subjects) - 1, args.n_pairs)):
        fixed_subj  = subjects[i]
        moving_subj = subjects[i + 1]

        # Find MRI and segmentation files
        fv = glob.glob(f'{fixed_subj}/PROCESSED/MPRAGE/T88_111/*_masked_gfc.img')
        mv = glob.glob(f'{moving_subj}/PROCESSED/MPRAGE/T88_111/*_masked_gfc.img')
        fs = glob.glob(f'{fixed_subj}/FSL_SEG/*_fseg.img')
        ms = glob.glob(f'{moving_subj}/FSL_SEG/*_fseg.img')

        if not all([fv, mv, fs, ms]):
            print(f"[Skip] Pair {i+1}: missing files")
            continue

        # Load data
        fixed_vol  = load_volume(fv[0], args.size).to(device)
        moving_vol = load_volume(mv[0], args.size).to(device)
        fixed_seg  = load_segmentation(fs[0], args.size)
        moving_seg = load_segmentation(ms[0], args.size)

        # Run inference
        with torch.no_grad():
            dvf, registered = model(moving_vol, fixed_vol, return_warped_source=True)

        # Warp segmentation
        warped_seg = warp_segmentation(moving_seg, dvf.cpu())

        # Compute DSC
        scores_before, mean_before = dice_score(moving_seg, fixed_seg)
        scores_after,  mean_after  = dice_score(warped_seg, fixed_seg)

        # Compute Jacobian
        jac_stats = jacobian_determinant_stats(dvf.cpu())

        results.append({
            'pair':f"P{i+1}",
            'dsc_before':mean_before,
            'dsc_after':mean_after,
            'scores_before':scores_before,
            'scores_after':scores_after,
            'neg_jac_pct':jac_stats['neg_pct'],
        })

        direction = '↑' if mean_after > mean_before else '↓'
        print(f"  Pair {i+1:2d}: DSC {mean_before:.3f} → {mean_after:.3f} "
              f"{direction} | Neg.Jacobian: {jac_stats['neg_pct']:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    mean_before = np.mean([r['dsc_before'] for r in results])
    mean_after  = np.mean([r['dsc_after']  for r in results])
    mean_neg_jac = np.mean([r['neg_jac_pct'] for r in results])

    print(f"Test pairs evaluated: {len(results)}")
    print(f"Mean DSC before: {mean_before:.3f}")
    print(f"Mean DSC after: {mean_after:.3f}")
    print(f"Improvement: {mean_after - mean_before:+.3f}")
    print(f"Mean Neg. Jacobian: {mean_neg_jac:.3f}%")
    print()
    print("Per-structure DSC:")
    for j, name in enumerate(label_names):
        b = np.mean([r['scores_before'][j] for r in results])
        a = np.mean([r['scores_after'][j]  for r in results])
        print(f"  {name:15s}: {b:.3f} → {a:.3f}  ({a-b:+.3f})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs('outputs/eval', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-pair DSC bar chart
    pairs = [r['pair'] for r in results]
    before = [r['dsc_before'] for r in results]
    after = [r['dsc_after']  for r in results]
    x = np.arange(len(pairs))
    w = 0.35

    axes[0].bar(x - w/2, before, w, label='Before', color='#E57373', alpha=0.8)
    axes[0].bar(x + w/2, after, w, label='After',  color='#4CAF50', alpha=0.8)
    axes[0].axhline(mean_before, color='red', linestyle='--', alpha=0.5, label=f'Mean before: {mean_before:.3f}')
    axes[0].axhline(mean_after, color='green', linestyle='--', alpha=0.5, label=f'Mean after: {mean_after:.3f}')
    axes[0].set_xlabel('Test Pair')
    axes[0].set_ylabel('Dice Score')
    axes[0].set_title('DSC per Test Pair: Before vs After Registration')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pairs, fontsize=7)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    # Per-label DSC bar chart
    before_per = [np.mean([r['scores_before'][j] for r in results]) for j in range(3)]
    after_per = [np.mean([r['scores_after'][j]  for r in results]) for j in range(3)]
    x2 = np.arange(3)

    axes[1].bar(x2 - w/2, before_per, w, label='Before', color='#E57373', alpha=0.8)
    axes[1].bar(x2 + w/2, after_per,  w, label='After', color='#4CAF50', alpha=0.8)
    for j in range(3):
        axes[1].annotate(
            f'{after_per[j]-before_per[j]:+.3f}',
            xy=(x2[j], max(before_per[j], after_per[j]) + 0.02),
            ha='center', fontsize=10, color='green', fontweight='bold')
    axes[1].set_xlabel('Brain Structure')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('DSC per Structure: Before vs After Registration')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(['CSF', 'Gray\nMatter', 'White\nMatter'])
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.suptitle(f'VoxelMorph Reproduction Results — OASIS-1 ({len(results)} test pairs)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_fig = 'outputs/eval/dsc_results.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Evaluate] Results figure saved: {out_fig}")

    return results

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VoxelMorph registration with DSC metric')
    parser.add_argument('--data_dir', default='data/raw/disc1', help='Directory containing OASIS subject folders')
    parser.add_argument('--weights', default='weights/voxelmorph_final.pt', help='Path to trained model weights')
    parser.add_argument('--n_pairs', type=int, default=20, help='Number of test pairs to evaluate')
    parser.add_argument('--size', type=int, default=128, help='Volume size used during training')
    args = parser.parse_args()

    evaluate(args)