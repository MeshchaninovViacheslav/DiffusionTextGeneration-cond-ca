from datasets import Dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from itertools import cycle
import gc
import torch
import numpy as np

disable_progress_bar()
set_verbosity_error()


class WikipediaDataset:
    def __init__(self, split, tokenizer, max_sequence_len, p_uncond=0):
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.p_uncond = p_uncond

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = self.dt.map(
            lambda element: conditional_preprocessing_wiki(
                element,
                self.tokenizer,
                self.max_sequence_len,
                self.p_uncond
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


def conditional_preprocessing_wiki(element, tokenizer, max_sequence_len, p_uncond=0):
    element["input_ids"] = element["input_ids"]
    element["length"] = sum(element["attention_mask"])

    n = min(max_sequence_len, element["length"])
    if np.random.rand() < p_uncond:
        ind = 0
    else:
        ind = np.random.randint(0, n)
    cond_ids = element["input_ids"][:ind]
    input_ids = element["input_ids"][ind:]

    cond_ = tokenizer.encode_plus(
        text=tokenizer.decode(cond_ids, skip_special_tokens=True),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    input_ = tokenizer.encode_plus(
        text=tokenizer.decode(input_ids, skip_special_tokens=True),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    output = {
        "input_ids": input_["input_ids"],
        "cond_ids": cond_["input_ids"],
        "input_mask": input_["attention_mask"],
        "cond_mask": cond_["attention_mask"],
    }
    return output
