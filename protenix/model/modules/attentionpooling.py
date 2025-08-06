import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, n_in: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # input: (L², n_in)

        attn_scores = self.attn_mlp(x).squeeze(-1)  # (L²,)
        attn_weights = F.softmax(attn_scores, dim=0)  # (L²,)
        pooled = torch.sum(attn_weights.unsqueeze(1) * x, dim=0)  # (n_in,)
        return pooled  # shape: (n_in,)
