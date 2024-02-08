import torch
import numpy as np
from transformers import AutoModel, T5EncoderModel, AutoTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, encoder_name, enc_normalizer,  is_change_sp_tokens=True):
        super().__init__()
        
        if "bert" in encoder_name.lower():
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu()
        elif "e5" in encoder_name.lower():
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu()
        elif "t5" in encoder_name.lower():
            self.encoder = T5EncoderModel.from_pretrained(encoder_name)
            self.embeddings = self.encoder.encoder.embed_tokens.weight.data.cpu()
        else:
            raise Exception("Unknown encoder name. Add encoder to ./model/encoder.py")
        
        self.enc_normalizer = enc_normalizer
        self.is_change_sp_tokens = is_change_sp_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        

    def forward(
            self,
            input_ids, attention_mask
    ):
        sequence_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)
        
        if self.is_change_sp_tokens:
            for sp_token_id in self.tokenizer.all_special_ids:
                sequence_output[input_ids == sp_token_id] = self._normalize_emb(self.embeddings[sp_token_id]).cuda()
        
        return sequence_output
    
    
    def _normalize_emb(self, x):
        return x / torch.norm(x) * np.sqrt(x.shape[-1])