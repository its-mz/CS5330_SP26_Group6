import torch
import voxelmorph as vxm


def build_voxelmorph_model():
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 32, 32, 32],
        integration_steps=7,
        device='cpu'
    )
    model.eval()
    return model


if __name__ == "__main__":
    model = build_voxelmorph_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model built ✓  Parameters: {total:,}")