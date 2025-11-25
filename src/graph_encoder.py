import torch
from torch import nn
from torch_geometric.nn import TransformerConv

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128,
                 n_heads=4, n_layers=2, dropout=0.1, pretrained_path=None,
                 use_lambdaKG=False):
        """
        Graph Transformer encoder:
        - Nếu use_lambdaKG=True, chỉ load pre-trained node embeddings
        - Nếu use_lambdaKG=False, tạo TransformerConv layers
        """
        super().__init__()
        self.use_lambdaKG = use_lambdaKG
        if use_lambdaKG:
            # placeholder, sẽ load LambdaKG embeddings external
            self.embeddings = None
        else:
            self.n_layers = n_layers
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.layers = nn.ModuleList()

            # first layer
            self.layers.append(TransformerConv(in_dim, hidden_dim // n_heads, heads=n_heads))
            # hidden layers
            for _ in range(n_layers - 2):
                self.layers.append(TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads))
            # last layer
            self.layers.append(TransformerConv(hidden_dim, out_dim // n_heads, heads=n_heads))

            # load pretrained nếu có
            if pretrained_path:
                try:
                    state_dict = torch.load(pretrained_path, map_location="cpu")
                    self.load_state_dict(state_dict, strict=False)
                    print(f"[GraphTransformerEncoder] Loaded pretrained weights from {pretrained_path}")
                except Exception as e:
                    print(f"[GraphTransformerEncoder] Failed to load pretrained weights: {e}")

    def forward(self, x, edge_index=None):
        if self.use_lambdaKG:
            # x = pre-loaded LambdaKG embeddings tensor
            return self.embeddings
        else:
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h, edge_index)
                if i < self.n_layers - 1:
                    h = self.relu(h)
                    h = self.dropout(h)
            return h

    def load_lambdaKG_embeddings(self, embeddings_tensor):
        """
        embeddings_tensor: [num_nodes, emb_dim]
        """
        self.embeddings = embeddings_tensor
        print(f"[GraphTransformerEncoder] LambdaKG embeddings loaded: {embeddings_tensor.shape}")
