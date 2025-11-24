import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    def __init__(self, entity2text=None, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.entity2text = entity2text or {}

    def encode(self, texts):
        # Returns torch.Tensor [batch, dim]
        embeddings = self.encoder.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        return embeddings

    def get_text(self, entity_id):
        if entity_id not in self.entity2text:
            raise ValueError(f"entity {entity_id} missing in entity2text mapping")
        return self.entity2text[entity_id]
