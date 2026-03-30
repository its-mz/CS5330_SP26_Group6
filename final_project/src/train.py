import os
import glob
import argparse
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import voxelmorph as vxm


# ── Dataset ───────────────────────────────────────────────────────────────────

class OASISDataset(Dataset):
    """
    OASIS-1 brain MRI dataset for pairwise registration.
    Each sample is a (moving, fixed) pair of consecutive volumes.
    Volumes are resized to target_size^3 for memory efficiency.
    """
    def __init__(self, data_dir: str, target_size: int = 128):
        self.vols = sorted(glob.glob(os.path.join(data_dir, '*.nii.gz')))
        self.size = target_size
        print(f"[Dataset] {len(self.vols)} volumes → {len(self.vols)-1} pairs "
              f"(resized to {target_size}³)")

    def __len__(self):
        return len(self.vols) - 1

    def __getitem__(self, idx):
        fixed  = self._load(self.vols[idx])
        moving = self._load(self.vols[idx + 1])
        return moving, fixed

    def _load(self, path: str) -> torch.Tensor:
        """Load NIfTI volume, normalize to [0,1], resize to target_size³."""
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        t = torch.tensor(vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
        t = F.interpolate(t, size=(self.size,) * 3,
                          mode='trilinear', align_corners=True)
        return t.squeeze(0)  # (1, size, size, size)


# ── Loss Functions ────────────────────────────────────────────────────────────

def ncc_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
             win: int = 9) -> torch.Tensor:
    """
    Normalized Cross-Correlation loss.
    Measures local intensity similarity between fixed and registered images.
    Returns negative NCC (to minimize).
    """
    I, J = y_true, y_pred
    win_size = win ** 3
    filt = torch.ones([1, 1, win, win, win], device=y_true.device) / win_size

    I2  = F.conv3d(I * I, filt, padding=win // 2)
    J2  = F.conv3d(J * J, filt, padding=win // 2)
    IJ  = F.conv3d(I * J, filt, padding=win // 2)
    Im  = F.conv3d(I,     filt, padding=win // 2)
    Jm  = F.conv3d(J,     filt, padding=win // 2)

    cc  = (IJ - Im * Jm) ** 2
    den = (I2 - Im ** 2) * (J2 - Jm ** 2)
    return -torch.mean(cc / (den + 1e-5))


def grad_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Smoothness regularization on displacement vector field.
    Penalizes large spatial gradients to ensure smooth, anatomically
    plausible deformations.
    """
    dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
    return torch.mean(dy) + torch.mean(dx) + torch.mean(dz)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # Dataset and dataloader
    dataset = OASISDataset(args.data_dir, target_size=args.size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 32, 32, 32],
        integration_steps=7,
        device=str(device)
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    loss_history = []

    print(f"\n[Train] Starting: {args.epochs} epochs × {len(loader)} pairs/epoch")
    print(f"[Train] Lambda regularization: {args.lam}")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for moving, fixed in loader:
            moving = moving.to(device)
            fixed  = fixed.to(device)

            optimizer.zero_grad()

            # Forward pass: predict DVF and registered image
            dvf, registered = model(moving, fixed, return_warped_source=True)

            # Combined loss: NCC similarity + gradient regularization
            loss = ncc_loss(fixed, registered) + args.lam * grad_loss(dvf)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        # Logging
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.1f}s/epoch")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir,
                                     f'voxelmorph_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.out_dir, 'voxelmorph_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\n[Train] Complete. Final model: {final_path}")

    # Plot and save loss curve
    os.makedirs('outputs/eval', exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (NCC + λ·Grad)')
    plt.title('VoxelMorph Training Loss on OASIS-1')
    plt.grid(True)
    loss_fig = 'outputs/eval/training_loss.png'
    plt.savefig(loss_fig, dpi=150)
    plt.close()
    print(f"[Train] Loss curve saved: {loss_fig}")

    return model, loss_history


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train VoxelMorph on OASIS-1 brain MRI dataset')
    parser.add_argument('--data_dir', default='data/train', help='Directory containing preprocessed .nii.gz volumes')
    parser.add_argument('--out_dir', default='weights', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--size', type=int, default=128, help='Isotropic volume size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--lam', type=float, default=0.5, help='Regularization weight lambda (default: 0.5)')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')
    args = parser.parse_args()

    train(args)