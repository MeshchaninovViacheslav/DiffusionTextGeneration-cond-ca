import torch
import numpy as np
from random import random
import torch.distributed as dist
from typing import List
from datasets import Dataset


def create_dataset(dataset_name):
    if dataset_name == "rocstory":
        return RocStoryDatasetDDP
    if dataset_name == "qqp":
        return QQPDatasetDDP


class RocStoryDatasetDDP:
    def __init__(self,
                split, tokenizer_cond, tokenizer_gen, max_sequence_len, max_context_len, train_path, valid_path):
        self.split = split
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_context_len = max_context_len
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

        self.dt = self.dt.with_format("pt", columns=["input_ids", "input_mask", "special_tokens_mask", "text_cond"])
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
            return_special_tokens_mask=True
        )

        output = {
            "input_ids": input_["input_ids"],
            "input_mask": input_["attention_mask"],
            "special_tokens_mask": input_["special_tokens_mask"],
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
        

class QQPDatasetDDP:
    def __init__(self,
                 split, tokenizer_cond=None, tokenizer_gen=None, max_sequence_len=None, max_context_len=None, base_path=None):
        self.split = split
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_context_len = max_context_len
        self.base_path = base_path
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0
        

    def split_data_across_gpu(self, dt: List[str]):
        if self.split == "train":
            indexes = np.random.default_rng(seed=self.epoch).permutation(len(dt))
        else:
            indexes = np.arange(len(dt))
        
        start_ind = self.device_id * (len(dt) // self.total_device_number)
        end_ind = (self.device_id + 1) * (len(dt) // self.total_device_number)
        if (self.device_id + 1) == self.total_device_number:
            indexes = indexes[start_ind:]
        else:
            indexes = indexes[start_ind:end_ind]
        
        dt = [dt[i] for i in indexes]
        return dt
    

    def load_data(self):
        labels = ['src', 'trg']

        texts = dict()
        for label in labels:
            with open(f'{self.base_path}/{self.split}/data.json', 'r') as f:
                texts[label] = []
                for line in f:
                    texts[label].append(eval(line)[label])

        dt = []
        for i in range(len(texts[labels[0]])):
            dt.append({
                'text_src': texts['src'][i],
                'text_trg': texts['trg'][i],
            })

        dt = self.split_data_across_gpu(dt)
        dt = Dataset.from_list(dt)

        self.dt = dt.map(
            self.batch_preprocessing_conditional,
            batched=True,
            load_from_cache_file=False,
            num_proc=30,
            desc="Dataset tokenization",
            batch_size=1000,
        )
        #self.dt = self.dt.with_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])
    
        return self.dt


    def batch_preprocessing_conditional(self, batch):
        # Text encode
        if self.split == 'train':
            swap_rate = 0.5
            blank_cond_rate = 0.1
        else:
            swap_rate = 0
            blank_cond_rate = 0
        
        if np.random.rand() < swap_rate:
            batch['text_trg'], batch['text_src'] = batch['text_src'], batch['text_trg']

        # input_ = self.tokenizer_gen(
        #     batch['text_trg'],
        #     add_special_tokens=True,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_sequence_len,
        # )

        if np.random.rand() < blank_cond_rate:
            batch['text_src'] = [''] * len(batch['text_src'])

        # cond_ = self.tokenizer_cond(
        #     batch['text_src'],
        #     add_special_tokens=True,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_sequence_len,
        # )
        output = {
            "text_src": batch["text_src"],
            "text_trg": batch["text_trg"],
        }

        return output
    

    def get_data(self):
        while True:
            yield self.load_data()
