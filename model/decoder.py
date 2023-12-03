import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class Decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=768, vocab_size=32100, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, hidden_size)
        self.act_fn = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    

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