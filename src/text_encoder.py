# src/text_encoder.py
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class TextEncoderPretrained(nn.Module):
    """
    Text Encoder cho HKA:
    - Dùng pretrained sentence-transformers
    - Output embedding cố định 128-d để match adapter
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", entity2text=None, out_dim=128):
        super().__init__()
        self.entity2text = entity2text or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = out_dim

        # Adapter layer để map hidden_state -> 128-d
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    @property
    def hidden_size(self):
        return self.out_dim

    def encode(self, texts, device="cpu"):
        """
        texts: list of strings
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        # mean pooling
        emb = last_hidden.mean(dim=1)
        # project to 128-d
        emb = self.proj(emb)
        return emb

    def get_text(self, entity_id):
        return self.entity2text[entity_id]
