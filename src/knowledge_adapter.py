# src/knowledge_adapter.py
import torch
from torch import nn
import torch.nn.functional as F

class KnowledgeAdapter(nn.Module):
    """
    Lightweight adapter for HKA: 2-layer FFN with ReLU.
    Input: graph embeddings
    Output: transformed embeddings, same dimension as input
    """
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
