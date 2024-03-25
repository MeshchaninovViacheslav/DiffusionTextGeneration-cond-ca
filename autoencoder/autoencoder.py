import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import AutoModel

from autoencoder.perceiver_resampler import PerceiverResampler

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(config.encoder.name)
        self.compressor = PerceiverResampler(
            num_latents=config.compressor.num_latents,
            embedding_dim=config.compressor.embedding_dim,
            latent_dim=config.compressor.latent_dim,
            hidden_size=config.compressor.hidden_size,
            n_heads=config.compressor.n_heads,
            n_layers=config.compressor.n_layers,
            max_seq_len=config.compressor.max_seq_len,
            dropout_p=config.compressor.dropout_p,
            ff_mult=config.compressor.ff_mult,
            qk_norm=config.compressor.qk_norm,
            std_init=config.compressor.std_init,
            latent_normalize=config.compressor.latent_normalize,
        )
        self.decoder = PerceiverResampler(
            num_latents=config.decoder.num_latents,
            embedding_dim=config.decoder.embedding_dim,
            latent_dim=config.decoder.latent_dim,
            hidden_size=config.decoder.hidden_size,
            n_heads=config.decoder.n_heads,
            n_layers=config.decoder.n_layers,
            max_seq_len=config.decoder.max_seq_len,
            dropout_p=config.decoder.dropout_p,
            ff_mult=config.decoder.ff_mult,
            qk_norm=config.decoder.qk_norm,
            std_init=config.decoder.std_init,
            latent_normalize=config.decoder.latent_normalize,
        )
        self.projector = nn.Linear(config.projector.hidden_size, config.projector.vocab_size, bias=False)

    @torch.no_grad()
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
            attention_mask=None,
        )
        
        logits = self.projector(recon_x)
        return logits
    
    def train(self):
        self.encoder.eval()

        self.decoder.train()
        self.compressor.train()
        self.projector.train()
        return self

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.compressor.eval()
        self.projector.eval()
        return self
    