import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, T5TokenizerFast, RobertaTokenizerFast, ElectraTokenizerFast

import sys

sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca")

from utils.util import dict_to_cuda, make_mask_wo_SEP_CLS

from data.create_dataset import create_rocstory_dataset, create_wiki_dataset
from data.dataset_clean_wiki import WikipediaCleanDatasetUnconditional

from model.roberta_encoder import RobertaEncoderModel
from model.electra_encoder import ElectraEncoderModel
from model.emb_encoder import EmbEncoderModel
from model.bert_encoder import BertEncoderModel


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
                output = encoder(**{
                        "input_ids": X["input_ids"],
                        "attention_mask": X["attention_mask"]
                    })

            mask = make_mask_wo_SEP_CLS(X["attention_mask"]) #torch.ones_like(X["attention_mask"])  # make_mask_wo_SEP_CLS(X["attention_mask"])
            output = output * mask[:, :, None]
            cur_sum = torch.sum(output, dim=[0, 1])
            cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])
            cur_num = torch.sum(mask).item()

            sum_ = cur_sum if sum_ is None else cur_sum + sum_
            sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
            num += cur_num

            mean_dif = (sum_ / num - cur_sum / cur_num)
            sqr_dif = (sqr_sum_ / num - cur_sqr_sum / cur_num)
            T.set_description(
                f"dif mean: {torch.sum(torch.abs(mean_dif)).item()}, dif std2: {torch.sum(torch.abs(sqr_dif)).item()}")
        if i == 10000:
            break

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    torch.save(mean, f'./data/encodings-{model_name}-{dataset_name}-mean.pt')
    torch.save(std, f'./data/encodings-{model_name}-{dataset_name}-std.pt')


if __name__ == "__main__":
    bert_cfg = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(bert_cfg)

    cfg = "bert-base-uncased"
    tokenizer_gen = BertTokenizerFast.from_pretrained(cfg)
    from model.bert_encoder_llm import BertEncoderModel
    encoder = torch.nn.DataParallel(BertEncoderModel.from_pretrained(
        "./lm_training/checkpoints/bert-training-768-120-0.15-None-2048-wiki_no_group/bert-150000/",
        enc_normalizer=None
    )).eval().cuda()

    max_sequence_len = 128
    batch_size = 2048
    train_dataset = next(WikipediaCleanDatasetUnconditional(
        split="train",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    ).get_data())

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
        model_name="my_bert-768-120-150000",
        dataset_name="wiki"
    )
