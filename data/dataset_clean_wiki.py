from datasets import Dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from itertools import cycle
import json
import gc
import numpy as np
from itertools import chain

disable_progress_bar()
set_verbosity_error()


class WikipediaCleanDataset:
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

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = self.dt.map(
            lambda element: conditional_preprocessing_wiki_text(
                element=element,
                tokenizer_bert=self.tokenizer_bert,
                tokenizer_cond=self.tokenizer_cond,
                tokenizer_gen=self.tokenizer_gen,
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
            test_path = "/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-clean_text/test/data-00000-of-00001.arrow"
            yield self.load_data(test_path)

        list_of_datasets = [
            f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-clean_text/train/data-{i:05d}-of-00004.arrow"
            for i in range(4)]
        for name_dt in cycle(list_of_datasets):
            yield self.load_data(name_dt)
            self.clear_data()


def conditional_preprocessing_wiki_text(
        element,
        tokenizer_bert, tokenizer_cond, tokenizer_gen,
        max_sequence_len: int = 96,
        pos_begin: float = 0.33,
        pos_end: float = 0.67,
):
    element = tokenizer_bert.encode_plus(element["sentence"])
    elem_count = sum(element["attention_mask"])
    delimeter_pos = int(
        (
                np.random.rand() * (pos_end - pos_begin) + pos_begin
        ) * elem_count
    )

    cond_ids = element["input_ids"][:delimeter_pos]
    input_ids = element["input_ids"][delimeter_pos:]

    cond_ = tokenizer_cond.encode_plus(
        text=tokenizer_bert.decode(cond_ids, skip_special_tokens=True),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    input_ = tokenizer_gen.encode_plus(
        text=tokenizer_bert.decode(input_ids, skip_special_tokens=True),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )
    #print(len(input_["input_ids"]))

    output = {
        "input_ids": input_["input_ids"],
        "cond_ids": cond_["input_ids"],
        "input_mask": input_["attention_mask"],
        "cond_mask": cond_["attention_mask"],
    }
    return output


class WikipediaCleanDatasetUnconditional:
    def __init__(
            self,
            split,
            tokenizer,
            max_sequence_len
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.text_column_name = "sentence"

    def tokenize_function(self, examples):
        return self.tokenizer(
            text=examples[self.text_column_name],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_len,
        )

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        block_size = self.max_sequence_len
        total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def load_data(self, path):
        self.dt = Dataset.from_file(path)
        self.dt = self.dt.map(
            self.tokenize_function,
            batched=True,
            num_proc=30,
            remove_columns=['sentence', 'score', '__index_level_0__'],
        )
        self.dt.set_format("pt", columns=["input_ids", "attention_mask"])
        return self.dt

    def clear_data(self):
        del self.dt
        gc.collect()

    def get_data(self):
        if self.split == "test":
            test_path = "/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-clean_text/test/data-00000-of-00001.arrow"
            for name_dt in cycle([test_path]):
                yield self.load_data(name_dt)
        if self.split == "train":
            list_of_datasets = [
                f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-clean_text/train/data-{i:05d}-of-00004.arrow"
                for i in range(4)]
            for name_dt in cycle(list_of_datasets):
                yield self.load_data(name_dt)
                self.clear_data()
