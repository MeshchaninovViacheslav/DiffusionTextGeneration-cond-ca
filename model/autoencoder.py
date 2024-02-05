import torch
import math
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.ffn = FeedForwardNetwork(input_dim=d_model,
                                      hidden_dim=4 * d_model,
                                      activation=nn.GELU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.ffn(self.pe[:x.size(1)])
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 activation: nn = nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.params = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(in_features=self.input_dim,
                      out_features=self.hidden_dim),
            activation(),
            nn.Linear(in_features=self.hidden_dim,
                      out_features=self.input_dim)
        )

    def forward(self,
                inputs: torch.Tensor):
        return self.params(inputs)


class AttentionLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_LM: int):
        super().__init__()
        self.MHA = MultiheadAttention(embed_dim=d_LM, 
                                      num_heads=num_heads,
                                      batch_first=True)
        self.FFN = FeedForwardNetwork(input_dim=d_LM,
                                      hidden_dim=d_LM * 4)
        self.norm1 = nn.LayerNorm(d_LM)
        self.norm2 = nn.LayerNorm(d_LM)
        
    def forward(self,
                latents: torch.Tensor,
                embeds: torch.Tensor = None,
                padding_mask: torch.Tensor = None):
        q = latents
        if embeds is not None:
            kv = torch.cat([latents, embeds], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([torch.ones(kv.size(dim=0),
                                                        q.size(dim=1),
                                                        device=padding_mask.device), padding_mask],
                                                        dim=1).type(torch.bool)

        else:
            kv = latents
        print(f'Q SIZE: {latents.size()}')
        print(f'KV SIZE: {kv.size()}')
        mha_out = self.norm1(self.MHA(query=q,
                     key=kv,
                     value=kv,
                     key_padding_mask=padding_mask)[0] + q)
        return self.norm2(mha_out + self.FFN(mha_out))


class CompressionNetwork(nn.Module):
    def __init__(self,
                 d_LM: int,
                 l_dim: int = 32,
                 d_ae: int = 64,
                 n_layers: int = 3,
                 num_heads: int = 12):
        super().__init__()
        self.l_dim = l_dim
        self.d_ae = d_ae
        self.attn_layers = nn.ModuleList([AttentionLayer(num_heads=num_heads,
                                                         d_LM=d_LM) for _ in range(n_layers)])
        self.linear = nn.Linear(in_features=d_LM,
                                out_features=d_ae)
        self.latents = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros((1, l_dim, d_LM))))
    
    def forward(self,
                embeds: torch.Tensor,
                padding_mask: torch.Tensor):
        # Broadcasting latents for torch.cat to work correctly
        latents = self.latents * torch.ones(embeds.size(dim=0), 1, 1).cuda() 
        print(f'LATENTS GRAD: {latents.requires_grad}')
        for attn_layer in self.attn_layers:
            latents = attn_layer(latents=latents,
                                 embeds=embeds,
                                 padding_mask=padding_mask)
            print(f'LATENTS GRAD COMP: {latents.requires_grad}')

        return F.normalize(self.linear(latents), dim=2) * math.sqrt(self.d_ae)
        

class ReconstructionNetwork(nn.Module):
    def __init__(self,
                 d_LM: int,
                 max_len: int,
                 l_dim: int = 32,
                 d_ae: int = 64,
                 n_layers: int = 3,
                 num_heads: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features=d_ae,
                                out_features=d_LM)
        self.attn_layers = nn.ModuleList([AttentionLayer(num_heads=num_heads,
                                                         d_LM=d_LM) for _ in range(n_layers)])
        self.pos_emb_ldim = PositionalEncoding(d_model=d_LM,
                                          max_len=l_dim)
        self.pos_emb_maxlen = PositionalEncoding(d_model=d_LM,
                                                 max_len=max_len)
        self.latents = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros((1, max_len, d_LM))))

    def forward(self,
                x,
                padding_mask: torch.Tensor = None):
        # Broadcasting the latents:
        latents = self.latents * torch.ones(x.size(dim=0), 1, 1,
                                            device=self.latents.device,
                                            requires_grad=False)

        print(f'LATENTS GRAD: {latents.requires_grad}')

        # latents = self.latents
        # latents += self.pos_emb_maxlen(latents)
        x = self.linear(x)
        # x += self.pos_emb_ldim(x)
        for attn_layer in self.attn_layers:
            x = attn_layer(latents=latents,
                           embeds=x)
            print(f'LATENTS GRAD COMP: {latents.requires_grad}')

        return x


class Autoencoder(nn.Module):
    def __init__(self,
                 d_LM: int,
                 max_len: int, 
                 d_ae: int = 64,
                 l_dim: int = 32,
                 n_layers: int = 3,
                 num_heads: int = 12):
        super().__init__()
        self.comp = CompressionNetwork(d_LM=d_LM,
                                       l_dim=l_dim,
                                       d_ae=d_ae,
                                       n_layers=n_layers,
                                       num_heads=num_heads)
        self.rec = ReconstructionNetwork(d_LM=d_LM,
                                         max_len=max_len,
                                         l_dim=l_dim,
                                         d_ae=d_ae,
                                         n_layers=n_layers,
                                         num_heads=num_heads)
        
    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None):
        c = self.comp(x, padding_mask)
        print(f'COMPRESSED GRAD {c.requires_grad}')
        r = self.rec(c, padding_mask)
        print(f'RECONSTRUCTED GRAD {r.requires_grad}')
        return r