import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    

class BertDecoder(nn.Module):
    def __init__(self, model_name, mode='mlm', is_cond=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased")
        bert_config.vocab_size = config.vocab_size
        bert_config.hidden_size = config.hidden_size

        if mode == 'transformer':
            bert_config.num_hidden_layers = 3
            bert_config.is_decoder = is_cond
            bert_config.add_cross_attention = is_cond
            self.bert = AutoModel.from_config(bert_config).encoder
            self.fc = nn.Linear(config.hidden_size, config.vocab_size)
            self.net = lambda **x: self.fc(self.bert(**x).last_hidden_state)

        elif mode == 'mlm':
            self.cls = BertOnlyMLMHead(config)
            self.net = lambda x: self.cls(x)
        else:
            print('Unknown decoder mode', flush=True)
            raise

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None):
        return self.net(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=self.get_extended_attention_mask(encoder_attention_mask),
        )
    
    def get_extended_attention_mask(self, attention_mask):
        if attention_mask is None:
            return None
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(attention_mask.dtype).min
        return extended_attention_mask