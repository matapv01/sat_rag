import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=256, out_dim=256, pretrained_path=None):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state_dict)
            print(f"Loaded pretrained GNN from {pretrained_path}")

    def forward(self, x, edge_index):
        h = self.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h
