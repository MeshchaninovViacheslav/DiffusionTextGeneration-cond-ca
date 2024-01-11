from datasets import Dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from itertools import cycle
import json
import gc
import torch
import numpy as np
import torch.distributed as dist
from typing import List
from datasets import load_from_disk

# disable_progress_bar()
# set_verbosity_error()

from data.preprocessing import glue_tokenize, glue_text_preprocessor, conditional_preprocessing_wiki
from data.dataset_clean_wiki import WikipediaCleanDataset


def create_dataset(dataset_name, downstream_task=None):
    if dataset_name == "wikipedia":
        return WikipediaDatasetDDP
    if dataset_name == "wikipedia-clean":
        return WikipediaCleanDataset
    if dataset_name == "rocstory":
        return RocStoryDatasetDDP
    if dataset_name == "glue":
        if downstream_task == "sst2":
            return SST2Dataset
    if dataset_name == "qqp":
        return QQPDatasetDDP
    

class QQPDatasetDDP:
    def __init__(self,
                 split, tokenizer_bert, tokenizer_cond, tokenizer_gen, max_sequence_len=50,
                 pos_begin: float = 0., pos_end: float = 0.67, is_conditional=True):
        self.split = split
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_cond_len = max_sequence_len
        self.pos_begin = pos_begin
        self.pos_end = pos_end
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0
        self.is_conditional = is_conditional
        

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
        base_path = f'/home/vmeshchaninov/shared_folder/qqp'

        texts = dict()
        for label in labels:
            with open(f'{base_path}/{self.split}/data.json', 'r') as f:
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
        self.dt = self.dt.with_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])  # , "cond_latent"])
    
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

        input_ = self.tokenizer_gen(
            batch['text_trg'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        if np.random.rand() < blank_cond_rate:
            batch['text_src'] = [''] * len(batch['text_src'])

        cond_ = self.tokenizer_cond(
            batch['text_src'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )
        output = {
            "input_ids": input_["input_ids"],
            "cond_ids": cond_["input_ids"],
            "input_mask": input_["attention_mask"],
            "cond_mask": cond_["attention_mask"],
        }

        return output
    

    def get_data(self):
        while True:
            yield self.load_data()


class WikipediaDatasetDDP:
    def __init__(self,
                 split, tokenizer_bert, tokenizer_cond, tokenizer_gen, max_sequence_len,
                 pos_begin: float = 0.33, pos_end: float = 0.67):
        self.split = split
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_cond_len = max_sequence_len
        self.pos_begin = pos_begin
        self.pos_end = pos_end
        self.device_number = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.number_of_datasets = 8

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = self.dt.map(
            self.batch_preprocessing,
            batched=True,
            load_from_cache_file=False,
            num_proc=30,
            desc="Dataset tokenization",
            batch_size=1000,
        )
        self.dt = self.dt.with_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])
        return self.dt

    def batch_preprocessing(self, batch):
        # Random split
        batch_size = len(batch["input_ids"])
        elem_counts = self.max_cond_len
        delimeter_poses = (
            (
                    np.random.rand(batch_size) *
                    (self.pos_end - self.pos_begin) + self.pos_begin
            ) * elem_counts
        ).astype(int)

        cond_ids_list = []
        input_ids_list = []
        for i, element_ids in enumerate(batch["input_ids"]):
            cond_ids_list.append(element_ids[:delimeter_poses[i]])
            input_ids_list.append(element_ids[delimeter_poses[i]:])

        # Tokens decode
        texts_cond = self.tokenizer_bert.batch_decode(cond_ids_list, skip_special_tokens=True)
        texts_input = self.tokenizer_bert.batch_decode(input_ids_list, skip_special_tokens=True)

        # Text encode
        cond_ = self.tokenizer_cond(
            texts_cond,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        input_ = self.tokenizer_gen(
            texts_input,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        output = {
            "input_ids": input_["input_ids"],
            "cond_ids": cond_["input_ids"],
            "input_mask": input_["attention_mask"],
            "cond_mask": cond_["attention_mask"],
        }
        # for key in output:
        #     output[key] = torch.tensor(output[key])
        return output

    def clear_data(self):
        del self.dt
        gc.collect()

    def get_data(self):
        if self.split == "valid":
            while True:
                test_path = "/home/vmeshchaninov/nlp_models/data/wikipedia/filtered_input_ids/test/data-00000-of-00001.arrow"
                yield self.load_data(test_path)
        elif self.split == "train":
            list_of_datasets = [
                f"/home/vmeshchaninov/nlp_models/data/wikipedia/filtered_input_ids/train/data-{i:05d}-of-{self.number_of_datasets:05d}.arrow"
                for i in range(self.number_of_datasets)]
            ind = self.device_number
            while True:
                yield self.load_data(list_of_datasets[ind])
                ind = (ind + self.total_device_number) % self.number_of_datasets
        else:
            raise Exception("Wrong data split")


class SST2Dataset:
    def __init__(self, split, tokenizer_bert, tokenizer_cond, tokenizer_gen, max_sequence_len):
        self.split = split
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.config = json.load(open("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/config.json", "rb"))

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = glue_text_preprocessor(self.dt, benchmark_name="sst2", config=self.config)
        self.dt = self.dt.map(
            lambda element: glue_tokenize(
                element,
                tokenizer_cond=self.tokenizer_cond,
                tokenizer_gen=self.tokenizer_gen,
                max_sequence_len=self.max_sequence_len,
            ),
            num_proc=30,
        )
        self.dt.set_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])
        return self.dt

    def get_data(self):
        if self.split == "valid":
            test_path = "/home/vmeshchaninov/nlp_models/data/glue/sst2/validation/dataset.arrow"
            yield self.load_data(test_path)

        if self.split == "test":
            test_path = "/home/vmeshchaninov/nlp_models/data/glue/sst2/test/dataset.arrow"
            yield self.load_data(test_path)

        if self.split == "train":
            list_of_datasets = [f"/home/vmeshchaninov/nlp_models/data/glue/sst2/train/dataset.arrow"]
            for name_dt in cycle(list_of_datasets):
                yield self.load_data(name_dt)


class RocStoryDatasetDDP:
    def __init__(self,
                 split, tokenizer_bert, tokenizer_cond, tokenizer_gen, max_sequence_len,
                 pos_begin: float = 0., pos_end: float = 0.67, is_conditional=True):
        self.split = split
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.max_cond_len = max_sequence_len
        self.pos_begin = pos_begin
        self.pos_end = pos_end
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0
        self.is_conditional = is_conditional
        self.dt = None

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
        dt = dt.select(indexes)
        return dt
    

    def load_data(self, path):
        if self.dt is not None:
            return self.dt

        dt = Dataset.from_file(path)
        dt = self.spilt_data_across_gpu(dt)

        self.dt = dt.map(
            self.batch_preprocessing_conditional if self.is_conditional else self.batch_preprocessing_unconditional,
            batched=True,
            load_from_cache_file=False,
            num_proc=30,
            desc="Dataset tokenization",
            batch_size=1000,
        )
        if self.is_conditional:
            self.dt = self.dt.with_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])
        else:
            self.dt = self.dt.with_format("pt", columns=["input_ids", "input_mask"])
        return self.dt

    def batch_preprocessing_conditional(self, batch):
        # Tokenize
        input_ids = batch["input_ids"]

        # Random split
        batch_size = len(batch["text"])
        elem_counts = self.max_cond_len
        delimeter_poses = (
            (
                    np.random.rand(batch_size) *
                    (self.pos_end - self.pos_begin) + self.pos_begin
            ) * elem_counts
        ).astype(int)

        cond_ids_list = []
        input_ids_list = []
        for i, element_ids in enumerate(input_ids):
            cond_ids_list.append(element_ids[:delimeter_poses[i]])
            input_ids_list.append(element_ids[delimeter_poses[i]:])

        skip_special_tokens = True
        # Tokens decode
        texts_cond = self.tokenizer_bert.batch_decode(cond_ids_list, skip_special_tokens=skip_special_tokens)
        texts_input = self.tokenizer_bert.batch_decode(input_ids_list, skip_special_tokens=skip_special_tokens)

        # Text encode
        cond_ = self.tokenizer_cond(
            texts_cond,
            add_special_tokens=skip_special_tokens,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        input_ = self.tokenizer_gen(
            texts_input,
            add_special_tokens=skip_special_tokens,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

        output = {
            "input_ids": input_["input_ids"],
            "cond_ids": cond_["input_ids"],
            "input_mask": input_["attention_mask"],
            "cond_mask": cond_["attention_mask"],
        }
        return output


    def batch_preprocessing_unconditional(self, batch):
        input_ = self.tokenizer_gen(
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

    def clear_data(self):
        del self.dt
        gc.collect()


    def get_data(self):
        if self.split == "valid":
            while True:
                test_path = "/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/test/data-00000-of-00001.arrow"
                yield self.load_data(test_path)
        elif self.split == "train":
            while True:
                train_path = "/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/train/data-00000-of-00001.arrow"
                yield self.load_data(train_path)
        else:
            raise Exception("Wrong data split")


class RocStoryDatasetEncodings:
    def __init__(self, split):
        self.split = split

    def load_data(self, path):
        dt = load_from_disk(path)
        dt = dt.with_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask", "gen_x", "cond_x"])
        return dt

    def get_data(self):
        if self.split == "valid":
            test_path = "/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/test_encodings/"
            return self.load_data(test_path)
        elif self.split == "train":
            train_path = "/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/train_encodings/"
            return self.load_data(train_path)
        else:
            raise Exception("Wrong data split")