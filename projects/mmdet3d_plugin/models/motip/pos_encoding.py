import torch
import torch.nn as nn

class Positional3DEncoding(nn.Module):
    """3D bbox + velocity → C-dim embedding.

    NOTE: despite the name, this is a small MLP encoder, not a sinusoidal
    positional encoding. Kept under this name for backwards compatibility
    with SparseBEV's checkpoint key naming.

    Input: (x,y,z,w,l,h,θ,vx,vy) = 9-dim
    """
    def __init__(self, embed_dim=256, input_dim=9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, bbox_3d, velocity):
        """
        bbox_3d: [N, 7] (x,y,z,w,l,h,θ)
        velocity: [N, 2] (vx,vy)
        Returns: [N, C]
        """
        state = torch.cat([bbox_3d, velocity], dim=-1)
        return self.mlp(state)