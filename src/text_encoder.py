from transformers import AutoModel, AutoTokenizer

class TextEncoderPretrained(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", entity2text=None):
        super().__init__()
        self.entity2text = entity2text or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts, device="cpu"):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        # mean pooling over last hidden state
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb

    def get_text(self, entity_id):
        return self.entity2text[entity_id]
