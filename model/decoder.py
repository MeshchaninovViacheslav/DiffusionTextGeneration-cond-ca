import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    

class BertDecoder(nn.Module):
    def __init__(self, mode='mlm'):
        super().__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased')
        if mode == 'transformer':
            config.num_hidden_layers = 3
            self.bert = AutoModel.from_config(config).encoder
            self.fc = nn.Linear(config.hidden_size, config.vocab_size)
            self.net = lambda x: self.fc(self.bert(x).last_hidden_state)

        elif mode == 'mlm':
            self.cls = BertOnlyMLMHead(config)
            self.net = lambda x: self.cls(x)
        else:
            print('Unknown decoder mode', flush=True)
            raise

    def forward(self, x):
        return self.net(x)