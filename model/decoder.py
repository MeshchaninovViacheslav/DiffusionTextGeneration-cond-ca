import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    

class BertDecoder(nn.Module):
    def __init__(self, model_name, mode='mlm'):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased")
        bert_config.vocab_size = config.vocab_size
        bert_config.hidden_size = config.hidden_size

        if mode == 'transformer':
            config.num_hidden_layers = 3
            self.bert = AutoModel.from_config(bert_config).encoder
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