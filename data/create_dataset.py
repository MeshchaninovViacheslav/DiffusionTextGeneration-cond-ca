import os
import json
import functools
import torch
import numpy as np
from datasets import load_dataset, load_from_disk, IterableDataset, disable_progress_bar, concatenate_datasets
from datasets.utils.logging import set_verbosity_error

from . import load

disable_progress_bar()
set_verbosity_error()


def roc_story_prep(x):
    return {"inputs": " ".join([x[f"sentence{i}"] for i in range(1, 6)])}


def create_rocstory_dataset(split, tokenizer, max_sequence_len):
    base_path = "/home/vmeshchaninov/nlp_models/data"
    if not os.path.isdir(f"{base_path}/wikipedia"):
        load.download_wikipedia(base_path)

    dt = load_dataset(
        path="adamlin/roc_story",
        ignore_verifications=True,
        split=split,
    )
    dt = dt.map(roc_story_prep, num_proc=32, remove_columns=dt.column_names)

    dt.set_transform(lambda x: tokenizer.encode_plus(
        x["inputs"][0],
        max_length=max_sequence_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    ))
    return dt


def create_conditional_dataset(dataset, split, tokenizer, max_sequence_len, p_uncond=0.5):
    if dataset == "rocstory":
        dt = load_dataset(
            path="adamlin/roc_story",
            ignore_verifications=True,
            split=split,
        )
        dt = dt.map(roc_story_prep, num_proc=32, remove_columns=dt.column_names)

    dt.set_transform(lambda element: conditional_preprocessing(element, tokenizer, max_sequence_len, p_uncond))
    return dt


def create_unconditional_dataset(dataset, split, tokenizer, max_sequence_len):
    if dataset == "rocstory":
        return create_rocstory_dataset(split, tokenizer, max_sequence_len)


def conditional_preprocessing(element, tokenizer, max_sequence_len, p_uncond=0):
    element = tokenizer.encode_plus(
        element["inputs_ids"][0],
        return_length=True,
        add_special_tokens=False,
    )

    n = min(max_sequence_len, element["length"][0])
    if np.random.rand() < p_uncond:
        ind = 0
    else:
        ind = np.random.randint(0, n - 1)
    cond_ids = element["input_ids"][:ind]
    input_ids = element["input_ids"][ind:]

    cond_ = tokenizer.encode_plus(
        text=tokenizer.decode(cond_ids),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    input_ = tokenizer.encode_plus(
        text=tokenizer.decode(input_ids),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    return {
        "input_ids": torch.tensor([input_["input_ids"]], dtype=torch.int64),
        "cond_ids": torch.tensor([cond_["input_ids"]], dtype=torch.int64),
        "input_mask": torch.tensor([input_["attention_mask"]], dtype=torch.int64),
        "cond_mask": torch.tensor([cond_["attention_mask"]], dtype=torch.int64),
    }
