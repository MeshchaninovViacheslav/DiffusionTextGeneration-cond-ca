import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys

sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca")

from utils.util import dict_to_cuda, make_mask_wo_SEP_CLS

from data.create_dataset import create_rocstory_dataset, create_wiki_dataset
from data.dataset_clean_wiki import WikipediaCleanDatasetUnconditional
from data.dataset import RocStoryDatasetDDP

from model.encoder_roberta import RobertaEncoderModel
from model.electra_encoder import ElectraEncoderModel
from model.emb_encoder import EmbEncoderModel
from model.encoder_bert import BertEncoderModel
from model.encoder_t5 import T5EncoderModel
from model.encoder_bart import BartEncoderModel

from create_config import create_config


def get_loader(tokenizer, max_sequence_len, batch_size):
    train_dataset = next(RocStoryDatasetDDP(
        split="train",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )
    return train_loader


def compute_mean_std(
        encoder,
        tokenizer,
        model_name, 
        dataset_name,
):
    sum_ = None
    sqr_sum_ = None
    num = 0

    max_sequence_len = 128
    batch_size = 2048

    train_loader = get_loader(
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
        batch_size=batch_size
    )
    T = tqdm(train_loader)

    for i, X in enumerate(T):
        X = dict_to_cuda(X)
        with torch.no_grad(), torch.autocast(device_type='cuda'):
            latent = encoder(**{
                    "input_ids": X["input_ids"],
                    "attention_mask": X["input_mask"]
                })
            
            embeddings = encoder.module.shared.weight.data.cpu()

            eos_emb = embeddings[tokenizer.eos_token_id].cuda()
            pad_emb = embeddings[tokenizer.pad_token_id].cuda()

            attention_mask = X["input_mask"]
            latent[torch.arange(len(latent)), attention_mask.sum(-1) - 1] = eos_emb
            latent[~attention_mask.bool()] = pad_emb

        mask = torch.ones_like(attention_mask)#make_mask_wo_SEP_CLS(X["input_mask"])
        latent = latent * mask[:, :, None]
        cur_sum = torch.sum(latent, dim=[0, 1])
        cur_sqr_sum = torch.sum(latent ** 2, dim=[0, 1])
        cur_num = torch.sum(mask).item()

        sum_ = cur_sum if sum_ is None else cur_sum + sum_
        sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
        num += cur_num

        mean = sum_[:3] / num
        std = torch.sqrt(sqr_sum_[:3] / num - mean ** 2)
        T.set_description(f"mean: {[m.item() for m in mean]}, std2: {[s.item() for s in std]}")

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    folder_path = f"/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/{dataset_name}/"
    os.makedirs(folder_path, exist_ok=True)
    torch.save(mean, f'{folder_path}/encodings-{model_name}-mean.pt')
    torch.save(std, f'{folder_path}/encodings-{model_name}-std.pt')


if __name__ == "__main__":
    config = create_config()
    cfg = config.model.encoder_name
    tokenizer = AutoTokenizer.from_pretrained(cfg)
    
    encoder = T5EncoderModel.from_pretrained(
        cfg,
        enc_normalizer=None
    ).eval()
    encoder = torch.nn.DataParallel(encoder).cuda()

    compute_mean_std(
        encoder,
        tokenizer, 
        model_name=config.model.encoder_name_hash,
        dataset_name="rocstory"
    )
