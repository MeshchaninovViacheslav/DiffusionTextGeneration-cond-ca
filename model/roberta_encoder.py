from transformers.models.roberta.modeling_roberta import (
    RobertaForMaskedLM as HuggingFaceRobertLMHeadModel
)


class RobertaEncoderModel(HuggingFaceRobertLMHeadModel):
    def __init__(self, config, enc_normalizer):
        super().__init__(config)
        self.enc_normalizer = enc_normalizer

    def forward(
            self,
            *args, **kwargs
    ):
        outputs = super().forward(
            *args, **kwargs, output_hidden_states=True
        )

        sequence_output = outputs.hidden_states[-3]
        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)
        return sequence_output
