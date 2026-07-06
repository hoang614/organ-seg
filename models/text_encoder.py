import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", output_dim=768):
        super().__init__()
        self.text_model = BertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.text_model.config.hidden_size, output_dim)

    def forward(self, text_inputs):
        text_outputs = self.text_model(**text_inputs)
        text_feats = text_outputs.last_hidden_state  # [B, T, hidden]
        text_feats = self.proj(text_feats)            # [B, T, output_dim]
        return text_feats