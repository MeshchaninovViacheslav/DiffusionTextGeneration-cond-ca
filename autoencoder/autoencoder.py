import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import AutoModel

from autoencoder.perceiver_resampler import PerceiverResampler

class AutoEncoder(nn.Module):
    def __init__(self,):
        super().__init__()

        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.compressor = PerceiverResampler(
            num_latents=32,
            embedding_dim=768,
            hidden_size=768,
            n_heads=12,
            n_layer=4,
        )
        self.decoder = PerceiverResampler(
            num_latents=64,
            embedding_dim=768,
            hidden_size=768,
            n_heads=12,
            n_layer=4,
        )
        self.projector = nn.Linear(768, 30522, bias=False)

    def encode(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def compress(self, encodings, attention_mask):
        return self.compressor(
            x=encodings,
            mask_x=attention_mask
        )
    
    def reconstruct(self, latents, attention_mask):
        return self.decoder(
            x=latents, 
            mask_x=None,
            mask_latent=attention_mask,
        )

    def forward(self, input_ids, attention_mask):
        encodings = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        latents = self.compress(
            encodings=encodings,
            attention_mask=attention_mask
        )
        
        recon_x = self.reconstruct(
            latents=latents,
            attention_mask=attention_mask,
        )
        
        logits = self.projector(recon_x)
        return logits