from datasets import Dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from itertools import cycle
import json
import gc

disable_progress_bar()
set_verbosity_error()

from data.preprocessing import glue_tokenize, glue_text_preprocessor, conditional_preprocessing_wiki


def create_dataset(dataset_name, downstream_task=None):
    if dataset_name == "wikipedia":
        return WikipediaDataset
    if dataset_name == "glue":
        if downstream_task == "sst2":
            return SST2Dataset


class WikipediaDataset:
    def __init__(self, split, tokenizer, max_sequence_len, pos_begin: float = 0.33, pos_end: float = 0.67):
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.pos_begin = pos_begin
        self.pos_end = pos_end

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = self.dt.map(
            lambda element: conditional_preprocessing_wiki(
                element=element,
                tokenizer=self.tokenizer,
                max_sequence_len=self.max_sequence_len,
                pos_begin=self.pos_begin,
                pos_end=self.pos_end,
            ),
            num_proc=30,
        )
        self.dt.set_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])
        return self.dt

    def clear_data(self):
        del self.dt
        gc.collect()

    def get_data(self):
        if self.split == "test":
            test_path = "/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/test/data-00000-of-00001.arrow"
            yield self.load_data(test_path)

        list_of_datasets = [f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/train/data-{i:05d}-of-00008.arrow"
                            for i in range(8)]
        for name_dt in cycle(list_of_datasets):
            yield self.load_data(name_dt)
            self.clear_data()


class SST2Dataset:
    def __init__(self, split, tokenizer, max_sequence_len):
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.config = json.load(open("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/config.json", "rb"))

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = glue_text_preprocessor(self.dt, benchmark_name="sst2", config=self.config)
        self.dt = self.dt.map(
            lambda element: glue_tokenize(
                element,
                tokenizer=self.tokenizer,
                max_sequence_len=self.max_sequence_len,
            ),
            num_proc=30,
        )
        self.dt.set_format("pt", columns=["input_ids", "cond_ids", "input_mask", "cond_mask"])
        return self.dt

    def get_data(self):
        if self.split == "test":
            test_path = "/home/vmeshchaninov/nlp_models/data/glue/sst2/validation/dataset.arrow"
            yield self.load_data(test_path)

        if self.split == "train":
            list_of_datasets = [f"/home/vmeshchaninov/nlp_models/data/glue/sst2/train/dataset.arrow"]
            for name_dt in cycle(list_of_datasets):
                yield self.load_data(name_dt)
