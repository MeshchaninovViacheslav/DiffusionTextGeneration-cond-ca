import torch
from torch import nn
from torch.nn import functional as F
import math


class PerceiverAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_heads, dropout_p=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim

        # initialize key, query, value matrix in one batch for optimization
        self.to_Q = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.to_KV = nn.Linear(embedding_dim, hidden_size * 2, bias = False)
        
        # output projection
        self.projector = nn.Linear(hidden_size, embedding_dim, bias = False) 
        
        # regularization using dropout
        self.attn_dropout = nn.Dropout(dropout_p) 
        self.proj_dropout = nn.Dropout(dropout_p)

    def forward(self, x, latents, mask_x, mask_latent):
        batch_size, seq_len, hidden_size = latents.size() # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch and split batch into three parts for q, k, v
        # move head forward to be the batch dim
        q = self.to_Q(latents)
        kv_input = torch.cat((x, latents), dim = 1)
        kv_mask = torch.cat((mask_x, mask_latent), dim=1)
        k, v = self.to_KV(kv_input).split(self.hidden_size, dim=2) 

        # Reshape [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, self.n_head, hidden_size // self.n_head]
        # Transpose [batch_size, seq_len, self.n_head, hidden_size // self.n_head] -> [batch_size, self.n_head, seq_len, hidden_size // self.n_head]
        # in order to calculate attention over different heads
        q = q.view(batch_size, q.shape[1], self.n_heads, hidden_size // self.n_heads).transpose(1, 2) 
        k = k.view(batch_size, k.shape[1], self.n_heads, hidden_size // self.n_heads).transpose(1, 2) 
        v = v.view(batch_size, v.shape[1], self.n_heads, hidden_size // self.n_heads).transpose(1, 2) 

        # Compute Attention scores
        # [batch_size, self.n_head, seq_len_q, seq_len_k]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        
        # Apply masking to attention scores, fill -inf to attention scores where mask is false 
        mask = kv_mask.view(kv_mask.shape[0], 1, kv_mask.shape[1]).repeat(1, mask_latent.shape[1], 1) * mask_latent.view(mask_latent.shape[0], mask_latent.shape[1], 1)
        mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2])
        #attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        mask = (1.0 - mask) * torch.finfo(attention_scores.dtype).min
        attention_scores = attention_scores + mask
        
        # Apply Softmax
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Add some regularization using dropout
        attention_scores = self.attn_dropout(attention_scores)

        # Calculate attention and resize to [batch_size, seq_len, hidden_size]
        y = attention_scores @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size) 

        # Apply output projection & dropout
        y = self.proj_dropout(self.projector(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_heads, dropout_p=0.1):
        super().__init__()
        
        self.ln_x = nn.LayerNorm(embedding_dim)
        self.ln_latents_1 = nn.LayerNorm(embedding_dim)
        self.attn = PerceiverAttention(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size, 
            n_heads=n_heads, 
            dropout_p=dropout_p,
        )

        self.ln_latents_2 = nn.LayerNorm(embedding_dim)

        self.proj_in = nn.Linear(embedding_dim, 4 * embedding_dim, bias = False)
        self.proj_out = nn.Linear(4 * embedding_dim, embedding_dim, bias = False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, latents, mask_x, mask_latent):
        x = self.ln_x(x)
        latents = self.ln_latents_1(latents)

        latents = latents + self.attn(x=x, latents=latents, mask_x=mask_x, mask_latent=mask_latent)
        latents = latents + self.dropout(self.proj_out(self.act(self.proj_in(self.ln_latents_2(latents)))))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        num_latents,
        embedding_dim,
        hidden_size,
        n_heads,
        n_layers,
        dropout_p=0.1,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embedding_dim))

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size, 
                    n_heads=n_heads, 
                    dropout_p=dropout_p,
                ) for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask_x=None, mask_latent=None):
        if mask_x is None:
            mask_x = torch.ones((x.shape[0], x.shape[1]), dtype=x.dtype)
        if mask_latent is None:
            mask_latent = torch.ones((x.shape[0], self.num_latents), dtype=x.dtype)
        
        latents = self.latents.view(1, self.num_latents, self.embedding_dim).repeat(x.shape[0], 1, 1)

        for layer in self.layers:
            latents = layer(x, latents, mask_x, mask_latent)

        return self.norm(latents)