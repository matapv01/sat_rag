# graphormer_encoder.py
import torch
from torch import nn
from transformers import GraphormerModel

class GraphormerEncoder(nn.Module):
    def __init__(self, model_name="clefourrier/graphormer-base-pcqm4mv2",
                 use_pretrained=True, device="cpu"):
        """
        Graphormer encoder cho HKA
        - model_name: HF checkpoint
        - use_pretrained: nếu True sẽ load pretrained HF weights
        """
        super().__init__()
        self.device = device
        self.use_pretrained = use_pretrained

        if use_pretrained:
            try:
                self.model = GraphormerModel.from_pretrained(model_name)
                print(f"[GraphormerEncoder] Loaded pretrained weights from {model_name}")
            except Exception as e:
                print(f"[GraphormerEncoder] Failed to load pretrained: {e}, fallback to random init")
                self.model = GraphormerModel.from_config(self._default_config())
        else:
            self.model = GraphormerModel.from_config(self._default_config())

        self.model.to(device)
        self.model.eval()  # default pretrained only

    def _default_config(self):
        from transformers import GraphormerConfig
        return GraphormerConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_encoder_layers=2,
        )

    def forward(self, node_feat, edge_index=None):
        """
        node_feat: [num_nodes, feat_dim]
        edge_index: [2, num_edges], optional
        """
        # HF Graphormer nhận input dạng dict
        inputs = {"node_feat": node_feat.to(self.device)}
        if edge_index is not None:
            inputs["attn_edge_type"] = None
            inputs["edge_input"] = edge_index.to(self.device)

        out = self.model(**inputs)
        return out.last_hidden_state  # [num_nodes, hidden_dim]

    def load_lambdaKG_embeddings(self, embeddings_tensor):
        """
        Nếu muốn dùng LambdaKG embeddings, override forward()
        """
        self.lambdaKG_embeddings = embeddings_tensor.to(self.device)
        self.use_lambdaKG = True
        print(f"[GraphormerEncoder] LambdaKG embeddings loaded: {embeddings_tensor.shape}")

    def forward_lambdaKG(self):
        if hasattr(self, "lambdaKG_embeddings"):
            return self.lambdaKG_embeddings
        else:
            raise ValueError("LambdaKG embeddings not loaded")
