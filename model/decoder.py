import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
    

class BertDecoder(nn.Module):
    def __init__(self, model_name, bert_config, mode='mlm'):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        bert_config.vocab_size = config.vocab_size
        bert_config.hidden_size = config.hidden_size
        bert_config.num_hidden_layers = 6
        self.bert = AutoModel.from_config(bert_config).encoder
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
        self.net = lambda x: self.fc(self.bert(x).last_hidden_state)

    def forward(self, x):
        return self.net(x)