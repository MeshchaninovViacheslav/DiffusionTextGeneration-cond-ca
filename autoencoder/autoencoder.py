import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import AutoModel

from autoencoder.perceiver_resampler import PerceiverResampler

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(config.encoder)
        self.compressor = PerceiverResampler(
            num_latents=config.compressor.num_latents,
            embedding_dim=config.compressor.embedding_dim,
            hidden_size=config.compressor.hidden_size,
            n_heads=config.compressor.n_heads,
            n_layers=config.compressor.n_layers,
            dropout_p=config.compressor.dropout_p,
        )
        self.decoder = PerceiverResampler(
            num_latents=config.decoder.num_latents,
            embedding_dim=config.decoder.embedding_dim,
            hidden_size=config.decoder.hidden_size,
            n_heads=config.decoder.n_heads,
            n_layers=config.decoder.n_layers,
            dropout_p=config.decoder.dropout_p,
        )
        self.projector = nn.Linear(config.projector.hidden_size, config.projector.vocab_size, bias=False)

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