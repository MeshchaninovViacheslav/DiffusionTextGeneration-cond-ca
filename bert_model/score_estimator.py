import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BaseModelOutputWithPastAndCrossAttentions, BertLayer, BertAttention, \
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
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.first_layers = torch.nn.ModuleList([BertLayer(config) for _ in range(0, self.num_hidden_layers // 2)])
        self.second_layers = torch.nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_hidden_layers // 2, self.num_hidden_layers)])
        # self.time_embedding = nn.Linear(self.hidden_size, self.hidden_size)
        self.projection_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size * 2, self.hidden_size) for _ in range(0, self.num_hidden_layers // 2)])
        self.time_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)])
        self.norm_layers = torch.nn.ModuleList(
            [torch.nn.LayerNorm(self.hidden_size, eps=self.config.layer_norm_eps) for _ in
             range(0, self.num_hidden_layers)])

    def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            emb_x=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        hidden_list = []

        for i, layer in enumerate(self.first_layers):
            hidden_list.append(hidden_state)
            hidden_state = self.norm_layers[i](hidden_state + self.time_layers[i](emb_t))
            layer_outputs = layer(
                hidden_state,
                attention_mask,
            )
            hidden_state = layer_outputs[0]

        for i, layer in enumerate(self.second_layers):
            res_state = hidden_list.pop()
            hidden_state = torch.cat((hidden_state, res_state), dim=2)
            hidden_state = self.projection_layers[i](hidden_state) + self.time_layers[i + self.num_hidden_layers // 2](
                emb_t)
            hidden_state = self.norm_layers[i + self.num_hidden_layers // 2](hidden_state)
            layer_outputs = layer(
                hidden_state,
                attention_mask,
            )
            hidden_state = layer_outputs[0]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_state,
        )


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
        self.LayerNorm = torch.nn.LayerNorm(self._hidden_layer_dim, eps=self.config.layer_norm_eps)
        #self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
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
            *args, **kwargs
    ):
        assert time_t is not None
        input_type = x_t.dtype
        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        # emb_t - shape [BATCH_SIZE; EMB_DIM]
        hidden_t = self.time_emb(emb_t)
        hidden_t = hidden_t[:, None, :]
        # emb_t - shape [BATCH_SIZE; EMB_DIM]
        # emb_t[:, None, :] - shape [BATCH_SIZE; 1; EMB_DIM] - for broadcasting only
        # SLAVIK LEGENDA

        emb_x = self.input_up_proj(x_t)

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_pos = self.position_embeddings(position_ids)

        emb = emb_x + emb_pos
        #emb = emb_x + emb_pos + hidden_t
        hidden_state = self.LayerNorm(emb)

        attention_mask = kwargs["attention_mask"]
        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=hidden_state.dtype
            )

        output = self.encoder(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            emb_x=emb_x,
        ).last_hidden_state

        output = self.output_down_proj(output).type(input_type)
        return output