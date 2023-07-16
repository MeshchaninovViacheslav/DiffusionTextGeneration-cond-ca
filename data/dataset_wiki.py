from datasets import Dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from itertools import cycle
import json
import gc
import numpy as np
from itertools import chain

disable_progress_bar()
set_verbosity_error()


class WikipediaDatasetUnconditional:
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
            test_path = "/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-text/test/data-00000-of-00001.arrow"
            for name_dt in cycle([test_path]):
                yield self.load_data(name_dt)
        if self.split == "train":
            list_of_datasets = [
                f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-text/train/data-{i:05d}-of-00008.arrow"
                for i in range(8)]
            for name_dt in cycle(list_of_datasets):
                yield self.load_data(name_dt)
                self.clear_data()
