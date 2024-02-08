import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys

sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca")

from utils.util import dict_to_cuda, make_mask_wo_SEP_CLS

from data import create_dataset

from model import Encoder

from create_config import create_config


def get_loader(config, tokenizer, batch_size):
    train_dataset = next(create_dataset(
        dataset_name=config.data.dataset_name,
    )(
        split="train",
        tokenizer_cond=tokenizer,
        tokenizer_gen=tokenizer,
        train_path=config.data.train_path,
        valid_path=config.data.valid_path,
        max_sequence_len=config.data.max_sequence_len + config.data.max_context_len,
        max_context_len=0,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )
    return train_loader


def compute_mean_std(
        config,
        encoder,
        tokenizer,
):
    sum_ = None
    sqr_sum_ = None
    num = 0

    batch_size = 2048

    train_loader = get_loader(
        config=config,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    T = tqdm(train_loader)

    for i, X in enumerate(T):
        X = dict_to_cuda(X)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = encoder(**{
                    "input_ids": X["input_ids"],
                    "attention_mask": X["input_mask"]
                })

        mask = 1 - X["special_tokens_mask"]
        output = output * mask[:, :, None]
        cur_sum = torch.sum(output, dim=[0, 1])
        cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])
        cur_num = torch.sum(mask).item()

        sum_ = cur_sum if sum_ is None else cur_sum + sum_
        sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
        num += cur_num

        mean = sum_[:3] / num
        std = torch.sqrt(sqr_sum_[:3] / num - mean ** 2)
        T.set_description(f"mean: {[m.item() for m in mean]}, std2: {[s.item() for s in std]}")

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    os.makedirs(config.data.dataset_path, exist_ok=True)
    torch.save(mean, config.data.enc_gen_mean)
    torch.save(std, config.data.enc_gen_std)


if __name__ == "__main__":
    config = create_config()

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    
    encoder = Encoder.from_pretrained(
        config.model.encoder_name,
        enc_normalizer=None,
        is_change_sp_tokens=False,
    ).cuda()
    encoder = torch.nn.DataParallel(encoder).cuda()

    compute_mean_std(
        config,
        encoder,
        tokenizer,
    )
