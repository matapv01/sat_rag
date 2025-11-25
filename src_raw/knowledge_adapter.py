import torch
from torch import nn

class KnowledgeAdapter(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, out_dim=128, dropout=0.1):
        """
        Knowledge Adapter 2-layer feed-forward, chuẩn paper:
        in_dim: input embedding dimension
        hidden_dim: hidden layer dimension
        out_dim: output embedding dimension (same as in_dim)
        """
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.ffn(x)
