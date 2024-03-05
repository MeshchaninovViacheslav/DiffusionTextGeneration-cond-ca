import os
import torch
import numpy as np
from random import random
import torch.distributed as dist
from typing import List
import datasets
from datasets import Dataset, load_from_disk
from itertools import cycle


class DatasetWikiDDP:
    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.base_path = config.data.dataset_path
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0
        self.files = self.get_files()
        self.iter_files = cycle(self.files)

    def get_files(self):
        path = f"{self.base_path}/{self.split}/"
        files = list(os.listdir(path))
        files = [t for t in files if ".arrow" in t]
        files = sorted(files, key = lambda f: int(f.split("-")[1]))
        return files

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
        
        return Dataset.from_dict(dt[indexes])

    def load_data(self):
        file = next(self.iter_files)
        path = f"{self.base_path}/{self.split}/{file}"
        dt = Dataset.from_file(path)
        dt = self.spilt_data_across_gpu(dt)

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
        if self.config.is_conditional:
            return self.batch_preprocessing_cond(batch)
        else:
            return self.batch_preprocessing_uncond(batch)

    def batch_preprocessing_uncond(self, batch):    
        return {"text_trg": batch["text"]}

    def batch_preprocessing_cond(self, batch):
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

    def get_data(self):
        while True:
            yield self.load_data()


