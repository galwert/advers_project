import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingProjector(nn.Module):
    def __init__(self, d_a, d_b, d_h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_a, d_h),
            nn.GELU(),
            nn.Linear(d_h, d_b)
        )

    def forward(self, x):
        # Ensure input dtype matches model weights
        x = x.to(self.net[0].weight.dtype)
        return F.normalize(self.net(x), dim=-1)


def loss_fn(proj, x, y, lam=0.1):
    # x: (B, d_a), y: (B, d_b)
    y = F.normalize(y, dim=-1)
    x_norm = F.normalize(x, dim=-1)

    out = proj(x)  # (B, d_b), already normalized

    # Alignment loss
    L_align = (1 - (out * y).sum(dim=-1)).mean()

    # Geometry loss
    sim_out = out @ out.T  # (B, B)
    sim_x = x_norm @ x_norm.T  # (B, B)
    L_geom = ((sim_out - sim_x) ** 2).mean()

    return L_align + lam * L_geom
