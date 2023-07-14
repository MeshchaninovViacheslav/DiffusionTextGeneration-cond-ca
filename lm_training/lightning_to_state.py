import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch

from lm_training.bert_lightning import BERTModel

bert = BERTModel.load_from_checkpoint(
    config=None,
    checkpoint_path="./checkpoints/bert-step_500000.ckpt"
)

#torch.save(bert.model.state_dict(), "../checkpoints/my_bert_pretrain.ckpt")

os.makedirs("./checkpoints/bert/", exist_ok=True)

bert.model.save_pretrained("./checkpoints/bert/")
