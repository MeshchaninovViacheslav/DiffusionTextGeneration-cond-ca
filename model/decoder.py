import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
    

class BertDecoder(nn.Module):
    def __init__(self, encoder_name, base_config):
        super().__init__()
        encoder_config = AutoConfig.from_pretrained(encoder_name)
        base_config.vocab_size = encoder_config.vocab_size
        base_config.hidden_size = encoder_config.hidden_size
        
        self.bert = AutoModel.from_config(base_config).encoder
        self.fc = nn.Linear(base_config.hidden_size, base_config.vocab_size)

    def forward(self, x):
        return self.fc(self.bert(x).last_hidden_state)
