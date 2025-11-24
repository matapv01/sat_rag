import torch
from torch import nn
import torch.nn.functional as F

class KnowledgeAdapter(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=256):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.ffn(x)
