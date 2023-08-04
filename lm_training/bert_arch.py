import torch
from typing import Dict, Optional
from transformers import BertPreTrainedModel

class BertLMPredictionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT, self).__init__(config)

        from transformers import BertModel

        self.encoder = BertModel(config, add_pooling_layer=False)
        self.projector = torch.nn.Linear(config.hidden_size, config.embedding_size)
        self.cls = BertLMPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return_dict = {}

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

        return_dict["last_hidden_state"] = outputs

        embeddings = self.projector(outputs)
        return_dict["embeddings"] = embeddings

        logits = self.cls(embeddings)
        return_dict["logits"] = logits
        return return_dict