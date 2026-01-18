# text_encoder.py
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class TextEncoderPretrained(nn.Module):
    """
    Vanilla Transformer text encoder (frozen).
    Output dim = hidden_size (expect 128).
    """
    def __init__(self, model_name="prajjwal1/bert-tiny", entity2text=None):
        super().__init__()
        self.entity2text = entity2text or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @property
    def hidden_size(self):
        return self.model.config.hidden_size  # should be 128 for bert-tiny

    def encode(self, texts, device="cpu", max_length=64):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb

