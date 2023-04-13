import json
import torch
import shutil
from itertools import cycle
import torch.distributed as dist
from copy import copy
import numpy as np
from datasets import load_dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error

from .preprocessing import text_preprocessor

disable_progress_bar()
set_verbosity_error()


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ex_iterables: dict, max_length, config, weights_sampling_mode="size"):
        self.ex_iterables: list = []
        self.weights: list = []
        for benchmark_name, dt in ex_iterables.items():
            if weights_sampling_mode == "size":
                if hasattr(dt, "num_rows"):
                    self.weights.append(dt.num_rows)
                else:
                    self.weights.append(config["data"][benchmark_name]["size"])
            else:
                self.weights.append(1)
            self.ex_iterables.append(dt)
        self.config = config
        self.max_length = max_length

    def __iter__(self):
        self.iterators = self._make_iters()
        self.current_weights = copy(self.weights)
        while True:
            index = torch.multinomial(input=torch.Tensor([self.current_weights]), num_samples=1).item()
            try:
                x = next(self.iterators[index])['inputs']
                yield x
            except StopIteration:
                self.current_weights[index] = 0
                if np.sum(self.current_weights) == 0:
                    self.iterators = self._make_iters()
                    self.current_weights = copy(self.weights)
                    return

    def _make_iters(self):
        return [iter(ex_iterable.shuffle()) for ex_iterable in self.ex_iterables]


class C4BatchDataset:
    def __init__(self, num_batch_data_files, tokenizer, max_sequence_len, project_config):
        self.num_batch_data_files = num_batch_data_files
        self.base_path = "/home/vmeshchaninov/nlp_models/data/c4/en"
        self.data_files = [f"{self.base_path}/c4-train.{idx:05d}-of-01024.json.gz" for idx in range(1024)]
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.config = self._set_config()
        if project_config.ddp:
            self.cache_dir = f"/home/vmeshchaninov/.cache/diffusion/{project_config.checkpoints_prefix}-{dist.get_rank()}"
        else:
            self.cache_dir = f"/home/vmeshchaninov/.cache/diffusion/{project_config.checkpoints_prefix}-{0}"
        self.batches_files = self._batch_data(self.data_files)

    def _batch_data(self, data_files, shuffle=True):
        new_len = len(data_files) // self.num_batch_data_files * self.num_batch_data_files
        data_files = data_files[:new_len]
        if shuffle:
            data_files = np.random.permutation(data_files[:new_len])
        return np.reshape(data_files, newshape=(-1, self.num_batch_data_files))

    def _prepare_dataset(self, data_files):
        self._clean_cache()

        dt = load_dataset(
            path=self.base_path,
            name="c4",
            data_files=data_files,
            ignore_verifications=True,
            split="train",
            cache_dir=self.cache_dir
        )
        dt = text_preprocessor(
            dt=dt,
            config=self.config,
            benchmark_name="c4"
        )
        dt.set_transform(lambda x: self.tokenizer(x["inputs"],
                                                  max_length=self.max_sequence_len,
                                                  padding="max_length",
                                                  truncation=True,
                                                  return_tensors="pt"))

        return dt

    def _set_config(self):
        config_path = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
        with open(config_path, "rb") as file:
            config = json.load(file)
        return config

    def __iter__(self):
        for data_files in self.batches_files:
            dt = self._prepare_dataset(data_files)
            yield dt

    def __len__(self):
        return len(self.batches_files)

    def _clean_cache(self):
        try:
            shutil.rmtree(self.cache_dir)
            print(f"Successfuly deleted {self.cache_dir}")
        except OSError:
            return
        pass


def roc_story_prep(x):
    return {"inputs": " ".join([x[f"sentence{i}"] for i in range(1, 6)])}


class RocStoryDataset:
    def __init__(self, tokenizer, max_sequence_len, split):
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.split = split
        self.dt = self._prepare_dataset(split)


    def _prepare_dataset(self, split):
        self.dt = load_dataset(
            path="adamlin/roc_story",
            ignore_verifications=True,
            split=split,
        )
        self.dt = self.dt.map(roc_story_prep, num_proc=32, remove_columns=self.dt.column_names)
        self.dt.set_transform(lambda x: self.tokenizer(x["inputs"],
                                                       max_length=self.max_sequence_len,
                                                       padding="max_length",
                                                       truncation=True,
                                                       return_tensors="pt"))

        return self.dt

    def __iter__(self):
        while True:
            yield self.dt

    def __len__(self):
        return 10_000
