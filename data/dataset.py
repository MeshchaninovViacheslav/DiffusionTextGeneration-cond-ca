from datasets import Dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from itertools import cycle
import json
import gc
import torch
import numpy as np
import torch.distributed as dist

# disable_progress_bar()
set_verbosity_error()

from data.preprocessing import glue_tokenize, glue_text_preprocessor, conditional_preprocessing_wiki
from data.dataset_clean_wiki import WikipediaCleanDataset


def create_dataset(dataset_name, downstream_task=None):
    if dataset_name == "wikipedia":
        return WikipediaDatasetDDP
    if dataset_name == "wikipedia-clean":
        return WikipediaCleanDataset
    if dataset_name == "glue":
        if downstream_task == "sst2":
            return SST2Dataset


class WikipediaDatasetDDP:
    def __init__(self,
                 split, tokenizer_bert, tokenizer_cond, tokenizer_gen, max_sequence_len,
                 pos_begin: float = 0.33, pos_end: float = 0.67):
        self.split = split
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_cond = tokenizer_cond
        self.tokenizer_gen = tokenizer_gen
        self.max_sequence_len = max_sequence_len
        self.pos_begin = pos_begin
        self.pos_end = pos_end
        self.device_number = dist.get_rank()
        self.total_device_number = dist.get_world_size()
        self.number_of_datasets = 8

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        # self.dt = self.dt.with_transform(
        #     self.batch_preprocessing,
        # )
        self.dt = self.dt.map(
            self.batch_preprocessing,
            batched=True,
            load_from_cache_file=False,
            num_proc=30,
            desc="Dataset tokenization"
        )
        return self.dt

    def batch_preprocessing(self, batch):
        # Random split
        elem_counts = np.sum(batch["attention_mask"], axis=1)
        delimeter_poses = (
            (
                    np.random.rand(elem_counts.shape[0]) *
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
        for key in output:
            output[key] = torch.tensor(output[key])
        return output

    def clear_data(self):
        del self.dt
        gc.collect()

    def get_data(self):
        if self.split == "valid":
            test_path = "/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/test/data-00000-of-00001.arrow"
            yield self.load_data(test_path)
        elif self.split == "train":
            list_of_datasets = [
                f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/train/data-{i:05d}-of-{self.number_of_datasets:05d}.arrow"
                for i in range(8)]
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
