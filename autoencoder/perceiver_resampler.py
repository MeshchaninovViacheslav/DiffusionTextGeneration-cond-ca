import torch
from torch import nn
from torch.nn import functional as F
import math


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


class PerceiverAttention(nn.Module):
    def __init__(self, embedding_dim, latent_dim, hidden_size, n_heads, dropout_p=0.1, qk_norm=True):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.dim_head = hidden_size // n_heads
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # normalization
        self.ln_x = nn.LayerNorm(embedding_dim)
        self.ln_latents = nn.LayerNorm(embedding_dim)

        # initialize key, query, value matrix in one batch for optimization
        if self.latent_dim != self.embedding_dim:
            self.latent_proj = nn.Linear(self.latent_dim, self.embedding_dim, bias = False)

        self.to_Q = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.to_KV = nn.Linear(embedding_dim, hidden_size * 2, bias = False)

        self.query_norm = RMSNorm(self.dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(self.dim_head) if qk_norm else nn.Identity()
        
        # output projection
        self.projector = nn.Linear(hidden_size, embedding_dim, bias = False) 
        
        # regularization using dropout
        self.attn_dropout = nn.Dropout(dropout_p) 
        self.proj_dropout = nn.Dropout(dropout_p)

    def forward(self, x, latents, mask_x, mask_latent):
        batch_size, seq_len, hidden_size = latents.size() # batch size, sequence length, embedding dimensionality

        # normalization
        x = self.ln_x(x)
        latents = self.ln_latents(latents)

        # calculate query, key, values for all heads in batch and split batch into three parts for q, k, v
        # move head forward to be the batch dim
        if self.latent_dim != self.embedding_dim:
            latents = self.latent_proj(latents)

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

        q = self.query_norm(q)
        k = self.query_norm(k)

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


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, mult=4, dropout_p=0.1):
        super().__init__()

        hidden_dim = int(embedding_dim * mult)
        self.ffd = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.ffd(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, latent_dim, hidden_size, n_heads, dropout_p=0.1, qk_norm=True, ff_mult=4):
        super().__init__()
        
        self.attn = PerceiverAttention(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size, 
            n_heads=n_heads, 
            dropout_p=dropout_p,
            qk_norm=qk_norm,
        )

        self.ffn = FeedForwardNetwork(embedding_dim=latent_dim, mult=ff_mult, dropout_p=dropout_p)

    def forward(self, x, latents, mask_x, mask_latent):
        latents = latents + self.attn(x=x, latents=latents, mask_x=mask_x, mask_latent=mask_latent)
        latents = latents + self.ffn(latents)
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        num_latents,
        embedding_dim,
        latent_dim,
        hidden_size,
        n_heads,
        n_layers,
        dropout_p=0.1,
        ff_mult=4,
        qk_norm=True,
        std_init=0.02,
        latent_normalize=False,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.latent_normalize = latent_normalize
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.normal_(self.latents, std = std_init)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    latent_dim=latent_dim,
                    hidden_size=hidden_size, 
                    n_heads=n_heads, 
                    dropout_p=dropout_p,
                    qk_norm=qk_norm,
                    ff_mult=ff_mult,
                ) for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask_x=None, mask_latent=None):
        if mask_x is None:
            mask_x = torch.ones((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        if mask_latent is None:
            mask_latent = torch.ones((x.shape[0], self.num_latents), dtype=x.dtype, device=x.device)
        
        latents = self.latents.view(1, self.num_latents, self.embedding_dim).repeat(x.shape[0], 1, 1)

        for layer in self.layers:
            latents = layer(x, latents, mask_x, mask_latent)
        
        if self.latent_normalize:
            latents = F.normalize(latents, dim=-1) * math.sqrt(latents.shape[-1])

        return latents