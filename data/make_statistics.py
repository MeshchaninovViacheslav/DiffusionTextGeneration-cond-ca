import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import create_dataset
from model import Encoder
from create_config import create_config


def get_loader(config, batch_size):
    train_dataset = next(create_dataset(
        config=config
    )(
        split="train",
        config=config,
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
        batch_size=batch_size
    )
    T = tqdm(train_loader)

    for i, batch in enumerate(T):
        trg = tokenizer(
            batch['text_trg'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=config.data.max_sequence_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )     

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = encoder(**{
                    "input_ids": trg["input_ids"].cuda(),
                    "attention_mask": trg["attention_mask"].cuda()
                })

        mask = 1 - trg["special_tokens_mask"].cuda()
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
    
    encoder = Encoder(
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
