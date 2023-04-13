import os
import json
import functools
import torch
import numpy as np
from datasets import load_dataset, load_from_disk, IterableDataset, disable_progress_bar, concatenate_datasets
from datasets.utils.logging import set_verbosity_error

from . import load
from .preprocessing import text_preprocessor, unsupervised_preprocessor, supervised_preprocessor
from .dataset import IterableDataset
from utils.util import dict_to_tensors

disable_progress_bar()
set_verbosity_error()


def create_c4_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data/c4/en"
    data_files = {
        "train": [f"{base_path}/c4-train.{idx:05d}-of-01024.json.gz" for idx in range(20)],
        # "validation": [f"{base_path}/c4-validation.{idx:05d}-of-00008.json.gz" for idx in range(8)],
        # "train": f"{base_path}/c4-train.*.json.gz",
        # "train": f"{base_path}/c4-validation.*.json.gz",
        "validation": f"{base_path}/c4-validation.*.json.gz",
    }
    return {"c4": load_dataset(path=base_path, data_files=data_files, split=split)}


def create_wiki_dpr_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data"
    if not os.path.isdir(f"{base_path}/wiki_dpr"):
        load.download_wiki_dpr(base_path)
    return {"wiki_dpr": load_from_disk(f"{base_path}/wiki_dpr").get(split)}


def create_glue_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data/glue/"
    configs = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte']
    return {config: load_from_disk(f"{base_path}/{config}").get(split) for config in configs}


def create_super_glue_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data/super_glue/"
    configs = ['copa', 'cb', 'boolq', 'wic', 'multirc']
    return {config: load_from_disk(f"{base_path}/{config}").get(split) for config in configs}


def create_unsupervised_dataset(split, config_path: str = "data/config.json", tokenizer=None,
                                max_sequence_len=128):
    with open(config_path, "rb") as file:
        config = json.load(file)
    #
    # datasets = dict()
    # datasets.update(create_c4_dataset(split))
    # # datasets.update(create_wiki_dpr_dataset(split))
    # # datasets.update(create_glue_dataset(split))
    # # datasets.update(create_super_glue_dataset(split))
    #
    # for name in list(datasets.keys()):
    #     dt = datasets[name]
    #     if dt is None:
    #         datasets.pop(name)
    #         continue
    #     datasets[name] = text_preprocessor(dt=dt, config=config, benchmark_name=name)
    #     if name not in ["c4"]:
    #         datasets[name] = unsupervised_preprocessor(datasets[name], benchmark_name=name)
    #     print(f"{name} is loaded")
    #
    # max_length = config["model"]["max_length"]
    # return IterableDataset(datasets, max_length=max_length, config=config, weights_sampling_mode=weights_sampling_mode)
    dt = create_c4_dataset(split)["c4"]
    dt = text_preprocessor(
        dt=dt,
        config=config,
        benchmark_name="c4"
    )
    dt.set_transform(lambda x: tokenizer(x["inputs"],
                                         max_length=max_sequence_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt"))
    return dt


def create_glue_unsupervised_dataset(split, tokenizer, max_sequence_len, downstream_task=None):
    datasets = dict()
    datasets.update(create_glue_dataset(split))
    datasets.update(create_super_glue_dataset(split))
    with open("/home/vmeshchaninov/DiffusionTextGeneration/data/config.json", "rb") as file:
        config = json.load(file)

    for name in list(datasets.keys()):
        dt = datasets[name]
        if dt is None or (downstream_task is not None and downstream_task != name):
            datasets.pop(name)
            continue
        print(f"{name} is loaded")
        datasets[name] = text_preprocessor(dt=dt, config=config, benchmark_name=name)
        datasets[name] = unsupervised_preprocessor(datasets[name], benchmark_name=name)
        datasets[name] = datasets[name]. \
            map(lambda x: tokenizer(x["inputs"],
                                    max_length=max_sequence_len,
                                    padding="max_length",
                                    truncation=False),
                num_proc=32,
                remove_columns=datasets[name].column_names). \
            filter(lambda x: np.sum(x["attention_mask"]) <= max_sequence_len)
    if downstream_task is None:
        dataset = concatenate_datasets(datasets.values())
    else:
        dataset = datasets[downstream_task]
    dataset.set_transform(lambda x: dict_to_tensors(x))
    return dataset


def create_glue_supervised_dataset(split, tokenizer, max_sequence_len, downstream_task=None):
    datasets = dict()
    datasets.update(create_glue_dataset(split))
    datasets.update(create_super_glue_dataset(split))
    with open("/home/vmeshchaninov/DiffusionTextGeneration/data/config.json", "rb") as file:
        config = json.load(file)

    for name in list(datasets.keys()):
        dt = datasets[name]
        if dt is None or (downstream_task is not None and downstream_task != name):
            datasets.pop(name)
            continue
        print(f"{name} is loaded")
        datasets[name] = text_preprocessor(dt=dt, config=config, benchmark_name=name)
        datasets[name] = supervised_preprocessor(datasets[name], benchmark_name=name)
        datasets[name] = datasets[name]. \
            map(lambda x: tokenizer(x["inputs"],
                                    max_length=max_sequence_len,
                                    padding="max_length",
                                    truncation=False),
                num_proc=32). \
            filter(lambda x: np.sum(x["attention_mask"]) <= max_sequence_len)
    if downstream_task is None:
        dataset = concatenate_datasets(datasets.values())
    else:
        dataset = datasets[downstream_task]
    dataset.set_transform(lambda x: dict_to_tensors(x))
    return dataset


def create_wikipedia_dataset(split, tokenizer, max_sequence_len):
    base_path = "/home/vmeshchaninov/nlp_models/data"
    if not os.path.isdir(f"{base_path}/wikipedia"):
        load.download_wikipedia(base_path)

    dt = load_from_disk(f"{base_path}/wikipedia").get(split)
    dt.set_transform(lambda x: tokenizer(x["inputs"],
                                         max_length=max_sequence_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt"))
    return dt


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

    def preprocessing(element):
        element = tokenizer.encode_plus(
            element["inputs"][0],
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
        # l = sum(cond_["attention_mask"])
        # cond_["input_ids"].pop(l - 1)
        # cond_["attention_mask"].pop(l - 1)

        input_ = tokenizer.encode_plus(
            text=tokenizer.decode(input_ids),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_len,
        )
        # input_["input_ids"].pop(0)
        # input_["attention_mask"].pop(0)

        return {
            "input_ids": torch.tensor([input_["input_ids"]], dtype=torch.int64),
            "cond_ids": torch.tensor([cond_["input_ids"]], dtype=torch.int64),
            "input_mask": torch.tensor([input_["attention_mask"]], dtype=torch.int64),
            "cond_mask": torch.tensor([cond_["attention_mask"]], dtype=torch.int64),
        }

    dt.set_transform(lambda element: preprocessing(element))
    return dt


def create_unconditional_dataset(dataset, split, tokenizer, max_sequence_len):
    if dataset == "rocstory":
        return create_rocstory_dataset(split, tokenizer, max_sequence_len)
