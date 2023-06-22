import pandas as pd
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import itertools

import sys

sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca")

from data.create_dataset import create_rocstory_dataset, create_wiki_dataset
from estimation_utils.metrics import RobertaMetric


def compute_roberta_score(
        train_loader,
        tokenizer_bert,
        metric_roberta_fn,
):
    T = tqdm(train_loader)
    result = dict()

    for i, X in enumerate(T):
        with torch.no_grad():
            texts = tokenizer_bert.batch_decode(X["input_ids"], skip_special_tokens=True)
            probs = metric_roberta_fn(texts)[1]
            for prob in probs:
                result[len(result)] = prob.item()
            with open("wikipedia-roberta_score.json", "w") as file:
                json.dump(result, file, indent=4)

            if len(result) > 10 ** 6:
                break


def filter_dataset(
        train_loader,
        tokenizer_bert,
        metric_roberta_fn,
):
    T = tqdm(train_loader)
    sentences = []
    scores = []
    num_chunk = 0

    for i, X in enumerate(T):
        with torch.no_grad():
            texts = tokenizer_bert.batch_decode(X["input_ids"], skip_special_tokens=True)
            probs = metric_roberta_fn(texts)[1].tolist()

            sentences.append(texts)
            scores.append(probs)

            if i % 5000 == 0:
                pd.DataFrame({
                    "sentence": list(itertools.chain(*sentences)),
                    "score": list(itertools.chain(*scores))
                }).to_csv(f"wikipedia-scored-{num_chunk:02d}.csv")
                num_chunk += 1
                sentences = []
                scores = []

    pd.DataFrame({
        "sentence": list(itertools.chain(*sentences)),
        "score": list(itertools.chain(*scores))
    }).to_csv(f"wikipedia-scored-{num_chunk:02d}.csv")


if __name__ == "__main__":
    bert_cfg = "bert-base-uncased"
    tokenizer_bert = BertTokenizerFast.from_pretrained(bert_cfg)

    metric_roberta_fn = RobertaMetric(device="cuda:0")

    batch_size = 1024
    train_dataset = create_wiki_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=30,
        pin_memory=True,
    )

    filter_dataset(
        train_loader,
        tokenizer_bert,
        metric_roberta_fn
    )
