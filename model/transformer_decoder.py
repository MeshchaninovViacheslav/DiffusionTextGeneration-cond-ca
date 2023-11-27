import torch
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Block, T5LayerNorm


t5_config = T5Config(**{
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "d_ff": 3072,
  #"d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "relu",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "relu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": True,
  "is_gated_act": False,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_heads": 8,
  "num_layers": 3,
  "output_past": True,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "vocab_size": 30522
})


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = torch.nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        hidden_states
    ):
        input_shape = hidden_states.size()[:2]
        batch_size, seq_length, _ = hidden_states.size()
        attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        for _, layer_module in enumerate(self.block):
            hidden_states = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
            )[0]


        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class Decoder(torch.nn.Module):
    def __init__(self, hidden_size=768, vocab_size=30522, layer_norm_eps=1e-12):
        super().__init__()
        self.decoder = T5Stack(config=t5_config)

        self.projector = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.projector.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.decoder(hidden_states)
        hidden_states = self.projector(hidden_states)
        return hidden_states