# graphormer_encoder.py
import torch
from torch import nn
from transformers import GraphormerModel

class GraphormerEncoder(nn.Module):
    def __init__(self, model_name="clefourrier/graphormer-base-pcqm4mv2",
                 use_pretrained=True, device="cpu"):
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

        self.embeddings = None  # placeholder
        self.edge_index = None

    def _default_config(self):
        from transformers import GraphormerConfig
        return GraphormerConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_encoder_layers=2,
        )

    def forward(self, node_feat=None, edge_index=None):
        """
        node_feat: [num_nodes, feat_dim]
        edge_index: [2, num_edges], optional
        """
        # Nếu đã gán embeddings, trả trực tiếp
        if self.embeddings is not None:
            return self.embeddings

        # Ngược lại, forward bình thường
        inputs = {"node_feat": node_feat.to(self.device)}
        if edge_index is not None:
            inputs["attn_edge_type"] = None
            inputs["edge_input"] = edge_index.to(self.device)

        out = self.model(**inputs)
        return out.last_hidden_state  # [num_nodes, hidden_dim]

    def load_lambdaKG_embeddings(self, embeddings_tensor):
        """
        Gán embeddings từ LambdaKG hoặc pretrained.
        """
        self.embeddings = embeddings_tensor.to(self.device)
        print(f"[GraphormerEncoder] Embeddings loaded: {embeddings_tensor.shape}")
