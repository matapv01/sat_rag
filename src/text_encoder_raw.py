import torch
from torch import nn

class TextEncoder(nn.Module):
    def __init__(self, entity2text=None, vocab_size=30522, max_len=64,
                 hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1,
                 pretrained_path=None):
        """
        Vanilla Transformer text encoder chuẩn paper:
        - hidden_dim=128
        - n_layers=2, n_heads=4
        - dropout=0.1
        - hỗ trợ load pretrained
        """
        super().__init__()
        self.entity2text = entity2text or {}
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        # output projection (mean pooling)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Load pretrained weights nếu có
        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                self.load_state_dict(state_dict, strict=False)
                print(f"[TextEncoder] Loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                print(f"[TextEncoder] Failed to load pretrained weights: {e}")

    def encode(self, texts, device="cpu"):
        """
        Encode list of texts -> tensor [batch, hidden_dim]
        """
        max_len = self.position_embedding.num_embeddings
        batch_size = len(texts)
        token_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            tokens = text.lower().split()[:max_len]
            token_ids[i, :len(tokens)] = torch.tensor(
                [hash(tok) % self.token_embedding.num_embeddings for tok in tokens],
                device=device
            )

        x = self.token_embedding(token_ids) + self.position_embedding(
            torch.arange(max_len, device=device)
        )
        x = self.dropout(x)

        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        x = x.mean(dim=1)        # mean pooling
        x = self.out_proj(x)
        return x

    def get_text(self, entity_id):
        if entity_id not in self.entity2text:
            raise ValueError(f"entity {entity_id} missing in entity2text mapping")
        return self.entity2text[entity_id]