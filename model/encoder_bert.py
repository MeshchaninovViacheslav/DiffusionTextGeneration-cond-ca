from torch import FloatTensor
from transformers.models.bert.modeling_bert import (
    BertLMHeadModel as HuggingFaceBertLMHeadModel, \
    BaseModelOutputWithPoolingAndCrossAttentions
)


class BertEncoderModel(HuggingFaceBertLMHeadModel):
    def __init__(self, config, enc_normalizer):
        super().__init__(config)
        self.enc_normalizer = enc_normalizer

    def forward(
            self,
            *args, **kwargs
    ):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            *args, **kwargs
        )

        sequence_output = outputs.last_hidden_state
        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)
        return sequence_output
