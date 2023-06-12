import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, T5TokenizerFast, RobertaTokenizerFast

import sys
sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca")

from utils.util import dict_to_cuda, make_mask_wo_SEP_CLS

from data.create_dataset import create_rocstory_dataset, create_wiki_dataset
from model.roberta_encoder import RobertaEncoderModel

def compute_mean_std(
        train_loader,
        encoder,
        tokenizer, tokenizer_gen,
        max_sequence_len,
        model_name, dataset_name,
):
    sum_ = None
    sqr_sum_ = None
    num = 0

    T = tqdm(train_loader)

    for i, X in enumerate(T):
        with torch.no_grad():
            text = tokenizer.batch_decode(X["input_ids"], skip_special_tokens=True)
            X = tokenizer_gen(
                text=text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_len,
                return_tensors="pt",
            )
            X = dict_to_cuda(X)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = encoder(**X)

            mask = make_mask_wo_SEP_CLS(X["attention_mask"])
            output = output * mask[:, :, None]
            cur_sum = torch.sum(output, dim=[0, 1])
            cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])

            sum_ = cur_sum if sum_ is None else cur_sum + sum_
            sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
            num += torch.sum(mask).item()

            mean = sum_[:3] / num
            std2 = sqr_sum_[:3] / num - mean ** 2
            T.set_description(f"mean: {mean.detach().cpu().tolist()}, std2: {std2.detach().cpu().tolist()}")
        if i == 1000:
            break

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    torch.save(mean, f'./data/encodings-{model_name}-{dataset_name}-mean.pt')
    torch.save(std, f'./data/encodings-{model_name}-{dataset_name}-std.pt')

if __name__ == "__main__":
    bert_cfg = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(bert_cfg)

    cfg = "roberta-base"
    tokenizer_gen = RobertaTokenizerFast.from_pretrained(cfg)
    encoder = RobertaEncoderModel.from_pretrained(
        cfg, enc_normalizer=None
    ).eval().cuda()

    max_sequence_len = 128
    batch_size = 512
    train_dataset = create_wiki_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )

    compute_mean_std(
        train_loader,
        encoder,
        tokenizer, tokenizer_gen,
        max_sequence_len,
        model_name="roberta_base",
        dataset_name="wiki"
    )

