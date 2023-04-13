import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BaseModelOutputWithPastAndCrossAttentions, BertAttention, \
    ACT2FN, apply_chunking_to_forward


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, add_cross_attention):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.crossattention = None
        if add_cross_attention:
            self.crossattention = BertAttention(config, position_embedding_type="absolute")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_outputs[0]
        if self.crossattention:
            cross_attention_outputs = self.crossattention(
                hidden_states=attention_output,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertBlock(torch.nn.Module):
    def __init__(self, config, is_input=False, is_output=False, is_middle=False):
        super().__init__()
        self.config = config
        if is_input or is_middle:
            self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.x_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        if is_output:
            self.norm = nn.LayerNorm(self.config.hidden_size * 2, eps=self.config.layer_norm_eps)
            self.x_proj = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.x_act_fn = ACT2FN[config.hidden_act]

        self.t_act_fn = torch.nn.SiLU()
        self.t_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.layer = BertLayer(config, add_cross_attention=is_input or is_output)

    def forward(self, x, t, attention_mask, cond=None, cond_mask=None):
        x = self.norm(x)
        x = self.x_act_fn(x)
        x = self.x_proj(x)

        t = self.t_act_fn(t)
        t = self.t_proj(t)

        hidden = x + t
        x = self.layer(
            hidden_states=hidden,
            attention_mask=attention_mask,
            encoder_hidden_states=cond,
            encoder_attention_mask=cond_mask
        )
        return x


class BertEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.input_blocks = torch.nn.ModuleList(
            [BertBlock(config, is_input=True) for _ in range(0, self.num_hidden_layers // 2)])
        self.middle_block = BertBlock(config, is_middle=True)
        self.output_blocks = torch.nn.ModuleList(
            [BertBlock(config, is_output=True) for _ in range(0, self.num_hidden_layers // 2)])

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            t_emb=None,
            cond=None,
            cond_mask=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x = block(x=x, attention_mask=attention_mask, t=t_emb, cond=cond, cond_mask=cond_mask)
            x_input_list.append(x)

        x = self.middle_block(x=x, attention_mask=attention_mask, t=t_emb)

        for _, block in enumerate(self.output_blocks):
            x = torch.cat([x, x_input_list.pop()], dim=2)
            x = block(x=x, attention_mask=attention_mask, t=t_emb, cond=cond, cond_mask=cond_mask)

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
        #hidden_t = emb_t
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
