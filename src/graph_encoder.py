import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=256, out_dim=256, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
