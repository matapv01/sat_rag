# src/text_encoder.py
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """
    Text encoder dùng SentenceTransformer / HuggingFace model.
    Cho phép trainable embeddings.
    entity2text mapping cần được truyền vào để lấy mô tả entity.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", entity2text=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entity2text = entity2text

    def get_text(self, entity):
        if self.entity2text is None:
            raise ValueError("entity2text mapping not provided")
        return self.entity2text[entity]

    @torch.no_grad()
    def encode(self, texts):
        """
        texts: list of strings
        return: [len(texts), hidden_dim]
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        output = self.encoder(input_ids=input_ids.to(self.encoder.device),
                              attention_mask=attention_mask.to(self.encoder.device))
        # mean pooling
        emb = (output.last_hidden_state * attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)).sum(1)
        emb = emb / attention_mask.sum(1, keepdim=True)
        return emb
