import torch
import numpy as np
import torch.distributed as dist
from typing import List
from datasets import Dataset


def create_dataset(dataset_name):
    if dataset_name == "rocstory":
        return RocStoryDatasetDDP


class RocStoryDatasetDDP:
    def __init__(self,
                 split, tokenizer, max_sequence_len):
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.max_cond_len = max_sequence_len
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
            self.batch_preprocessing_unconditional,
            batched=True,
            load_from_cache_file=False,
            num_proc=30,
            desc="Dataset tokenization",
            batch_size=1000,
        )
        
        self.dt = self.dt.with_format("pt", columns=["input_ids", "input_mask"])
        return self.dt


    def batch_preprocessing_unconditional(self, batch):
        input_ = self.tokenizer(
            batch["text"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        output = {
            "input_ids": input_["input_ids"],
            "input_mask": input_["attention_mask"],
        }
        return output


    def get_data(self):
        if self.split == "valid":
            while True:
                test_path = "/home/vmeshchaninov/nlp_models/data/rocstories/validation/data.txt"
                yield self.load_data(test_path)
        elif self.split == "train":
            while True:
                train_path = "/home/vmeshchaninov/nlp_models/data/rocstories/train/data.txt"
                yield self.load_data(train_path)
        else:
            raise Exception("Wrong data split")
