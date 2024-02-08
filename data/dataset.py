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
                split, tokenizer_cond, tokenizer_gen, max_sequence_len, max_context_len, base_path):
        self.split = split
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_context_len = max_context_len
        self.base_path = base_path
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
    

    def load_data(self):
        path = f"{self.base_path}/{self.split}/data.txt"
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
        return self.dt


    def batch_preprocessing(self, batch):
        # Random split
        if self.split == 'train':
            blank_cond_rate = 0.1
        else:
            blank_cond_rate = 0

        texts_cond = []
        texts_input = []
        for i, text in enumerate(batch["text"]):
            sentences = self.splitOnChars(text)
            if random() < blank_cond_rate:
                texts_cond.append("")
            else:
                texts_cond.append(sentences[0])
            texts_input.append("".join(sentences[1:]))

        output = {
            "text_src": batch["text_src"],
            "text_trg": batch["text_trg"],
        }
        return output
    
    def splitOnChars(self, text, chars):
        answer = []
        start = 0
        for i, char in enumerate(text):
            if char in chars:
                answer.append(text[start:i+1])
                start = i+1
        answer.append(text[i+1:])
        return answer

    def get_data(self):
        if self.split == "valid":
            while True:
                yield self.load_data()
        elif self.split == "train":
            while True:
                yield self.load_data()
        else:
            raise Exception("Wrong data split")
