import sys
import torch

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from lm_training.bert_arch import BERT


class BertEncoderModel(BERT):
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

        sequence_output = outputs["embeddings"].type(dtype=torch.float)
        if self.enc_normalizer is not None:
            sequence_output = self.enc_normalizer.normalize(sequence_output)
        return sequence_output
