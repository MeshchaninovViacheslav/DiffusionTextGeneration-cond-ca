from transformers.models.bart.modeling_bart import (
    BartEncoder as HuggingFaceT5EncoderModel,
)


class BartEncoderModel(HuggingFaceT5EncoderModel):
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
        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)

        return sequence_output