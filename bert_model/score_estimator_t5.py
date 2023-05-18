import torch
import math
from typing import Optional


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ScoreEstimatorEMB(torch.nn.Module):
    def __init__(self, input_size, config):
        super(ScoreEstimatorEMB, self).__init__()

        self.input_size = input_size
        self.config = config
        hidden_layer_dim = self.config.hidden_size
        self._hidden_layer_dim = hidden_layer_dim
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
        )

        self.encoder = BertEncoder(config)

        self._max_position_embeddings = self.config.max_position_embeddings
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

        self.input_up_proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, self._hidden_layer_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._hidden_layer_dim, self._hidden_layer_dim)
        )

        self.output_down_proj = torch.nn.Sequential(
            torch.nn.Linear(self._hidden_layer_dim, self._hidden_layer_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._hidden_layer_dim, input_size)
        )

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            cond=None,
            *args, **kwargs
    ):
        assert time_t is not None
        input_type = x_t.dtype

        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        hidden_t = self.time_emb(emb_t)
        # hidden_t = emb_t
        hidden_t = hidden_t[:, None, :]

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_pos = self.position_embeddings(position_ids)

        emb_x = self.input_up_proj(x_t)
        hidden_state = emb_x + emb_pos

        # emb = emb_x + emb_pos + hidden_t

        attention_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        cond_mask = kwargs["cond_mask"] if "cond_mask" in kwargs else None

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=hidden_state.dtype
            )
        if cond_mask is not None:
            cond_mask = self.get_extended_attention_mask(
                attention_mask=cond_mask,
                dtype=hidden_state.dtype
            )

        output = self.encoder(
            x=hidden_state,
            attention_mask=attention_mask,
            t_emb=hidden_t,
            cond=cond,
            cond_mask=cond_mask,
        )

        output = self.output_down_proj(output).type(input_type)
        return output
