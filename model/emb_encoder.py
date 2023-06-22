from transformers.models.bert.modeling_bert import (
    BertLMHeadModel as HuggingFaceBertLMHeadModel
)


class EmbEncoderModel(HuggingFaceBertLMHeadModel):
    def __init__(self, config, enc_normalizer):
        super().__init__(config)
        self.enc_normalizer = enc_normalizer

    def forward(
            self,
            *args, **kwargs
    ):
        sequence_output = self.bert.embeddings.word_embeddings(
            kwargs["input_ids"]
        )
        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)
        return sequence_output
