from transformers.models.t5.modeling_t5 import (
    BaseModelOutputWithPastAndCrossAttentions, \
    T5EncoderModel as HuggingFaceT5EncoderModel,
)


class T5EncoderModel(HuggingFaceT5EncoderModel):
    def __init__(self, config, enc_normalizer):
        super().__init__(config)
        self.enc_normalizer = enc_normalizer

    def forward(
            self,
            *args, **kwargs
    ):
        outputs = super().forward(
            *args, **kwargs
        )

        sequence_output = outputs.last_hidden_state
        normed = self.enc_normalizer.normalize(sequence_output)

        return normed