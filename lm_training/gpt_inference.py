import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import json
import torch
import ml_collections
from transformers import AutoTokenizer
from data.dataset import create_dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning import seed_everything, Trainer

from lm_training.dataset_lightning import WikiDataModule
from lm_training.gpt_lightning import GPTModel


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6
    optim.precision = "16"

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 100_000
    training.eval_freq = 50_000
    training.batch_size = 512 // torch.cuda.device_count()

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 128
    data.num_workers = 16
    data.pos_begin = 0.33
    data.pos_end = 0.67

    config.project_name = "lm-training"
    config.exp_name = "gpt2-training"
    config.seed = 0

    return config


def get_model(config):
    gpt = GPTModel.load_from_checkpoint(config=config, checkpoint_path="./checkpoints/gpt2-training/step_500000.ckpt")
    return gpt


def get_dataloader(config, tokenizer, batch_size):
    bert_cfg = "bert-base-uncased"
    tokenizer_bert = BertTokenizerFast.from_pretrained(bert_cfg)

    valid_dataset = next(create_dataset(
        dataset_name="wikipedia-clean",
    )(
        split="test",
        tokenizer_bert=tokenizer_bert,
        tokenizer_cond=tokenizer,
        tokenizer_gen=tokenizer,
        max_sequence_len=config.data.max_sequence_len,
        pos_begin=config.data.pos_begin,
        pos_end=config.data.pos_end,
    ).get_data())
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return valid_dataloader


def main():
    texts_path = "./generated_texts"
    model_name = "gpt2-500_000"
    os.makedirs(texts_path, exist_ok=True)
    file_name = f"{texts_path}/{model_name}.json"

    num_gen_texts = 8192
    batch_size = 256

    config = create_config()
    seed_everything(config.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = 50256

    gpt = get_model(config)
    dataloader = get_dataloader(config, tokenizer, batch_size)

    result_texts = []

    for inputs in dataloader:
        cond_inputs = {"input_ids": inputs["cond_ids"], "attention_mask": inputs["cond_mask"]}
        outputs = gpt.generate_text(inputs=cond_inputs, max_new_tokens=64)

        cond_texts = tokenizer.batch_decode(inputs["cond_ids"], skip_special_tokens=True)
        gt_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

        for i in range(len(outputs)):
            prompt_length = len(
                cond_texts[i]
            )
            full_text = tokenizer.decode(
                outputs[i],
                skip_special_tokens=True
            )
            text = full_text[prompt_length:]
            result_texts.append(
                {
                    "CONDITION": cond_texts[i],
                    "GEN": text,
                    "GT": gt_texts[i],
                    "FULL": full_text
                }
            )

        json.dump(result_texts, open(file_name, "w"), indent=4)
        if len(result_texts) >= num_gen_texts:
            break

    print(file_name)


main()
