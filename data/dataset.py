import torch
import numpy as np
from random import random
import torch.distributed as dist
from typing import List
from datasets import Dataset


def create_dataset(dataset_name):
    if dataset_name == "rocstory":
        return RocStoryDatasetDDP


class RocStoryDatasetDDP:
    def __init__(self,
                split, tokenizer_cond, tokenizer_gen, max_sequence_len, max_cond_len, train_path, valid_path):
        self.split = split
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_cond_len = max_cond_len
        self.train_path = train_path
        self.valid_path = valid_path
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0


    def spilt_data_across_gpu(self, dt: List[str]):
        if self.split == "train":
            indexes = np.random.default_rng(seed=self.epoch).permutation(len(dt))
        else:
            indexes = np.arange(len(dt))
        
        start_ind = self.device_id * (len(dt) // self.total_device_number)
        end_ind = (self.device_id + 1) * (len(dt) // self.total_device_number)
        if (self.device_id + 1) == self.total_device_number:
            indexes = indexes[start_ind:]
        else:
            indexes = indexes[start_ind: end_ind]
        
        dt = [dt[i] for i in indexes]
        return dt
    

    def load_data(self, path):
        dt = []
        with open(path, "r") as file:
            for l in file:
                dt.append(l.strip())
        dt = self.spilt_data_across_gpu(dt)
        dt = Dataset.from_list([{"text": t} for t in dt])

        self.dt = dt.map(
            self.batch_preprocessing,
            batched=True,
            load_from_cache_file=False,
            num_proc=30,
            desc="Dataset tokenization",
            batch_size=1000,
        )

        self.dt = self.dt.with_format("pt", columns=["input_ids", "input_mask", "text_cond"])
        return self.dt


    def batch_preprocessing(self, batch):
        # Random split
        batch_size = len(batch["text"])

        texts_cond = []
        texts_input = []
        for i, text in enumerate(batch["text"]):
            sent = text.split(".")
            if random() < 0.1:
                texts_cond.append("")
            else:
                texts_cond.append(sent[0])
            texts_input.append(".".join(sent[1:]))

        # Text encode
        input_ = self.tokenizer_gen(
            texts_input,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        output = {
            "input_ids": input_["input_ids"],
            "input_mask": input_["attention_mask"],
            "text_cond": texts_cond,
        }
        return output


    def get_data(self):
        if self.split == "valid":
            while True:
                yield self.load_data(self.valid_path)
        elif self.split == "train":
            while True:
                yield self.load_data(self.train_path)
        else:
            raise Exception("Wrong data split")
