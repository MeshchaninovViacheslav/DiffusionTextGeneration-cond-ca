import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union

from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF, T5Config

t5_config = T5Config(**{
    "d_ff": 3072,
    "d_kv": 64,
    "d_model": 768,
    "decoder_start_token_id": 0,
    "dense_act_fn": "relu",  # "gelu", #"relu",
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "relu",
    "initializer_factor": 1.0,  # 0.02, #1.0,
    "layer_norm_epsilon": 1e-6,  # 1e-12, #1e-6,
    "model_type": "t5",
    "n_positions": 512,
    "num_decoder_layers": 12,
    "num_heads": 12,
    "num_layers": 12,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "is_decoder": True,
})


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        config = t5_config
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self_attention_outputs[0]

        # clamp inf values to enable fp16 training
        hidden_states = self.clamp_value(hidden_states)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            hidden_states = self.clamp_value(hidden_states)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        hidden_states = self.clamp_value(hidden_states)

        return hidden_states

    def clamp_value(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


TransformerBlock = T5Block


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = 6
        self.hidden_size = 768
        self.input_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.time_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
        )
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            cond=None,
            cond_mask=None,
    ):

        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            x = x + self.time_layers[i](emb_t)
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask
            )

        for i, block in enumerate(self.output_blocks):
            x = x + x_input_list.pop() + self.time_layers[i + self.num_hidden_layers // 2](emb_t)
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask
            )
        x = self.LayerNorm(x)

        return x


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


class ScoreEstimatorEMB(nn.Module):
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

        self.encoder = TransformerEncoder(config)

        self._max_position_embeddings = self.config.max_position_embeddings
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

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

        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        hidden_t = self.time_emb(emb_t)
        # hidden_t = emb_t
        hidden_t = hidden_t[:, None, :]

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_pos = self.position_embeddings(position_ids)

        emb_x = x_t
        hidden_state = emb_x + emb_pos

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
            emb_t=hidden_t,
            cond=cond,
            cond_mask=cond_mask,
        )
        return output
