import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import lightning as L
from torch.utils.data import DataLoader

from data.dataset_clean_wiki import WikipediaCleanDatasetUnconditional


class WikiDataModule(L.LightningDataModule):
    def __init__(self, tokenizer, config, collate_fn=None, ):
        super(WikiDataModule, self).__init__()

        self.max_sequence_len = config.data.max_sequence_len
        self.batch_size = config.training.batch_size
        self.collate_fn = collate_fn
        self.num_workers = config.data.num_workers

        self.train_dataset_iter = WikipediaCleanDatasetUnconditional(
            split="train",
            tokenizer=tokenizer,
            max_sequence_len=self.max_sequence_len,
        ).get_data()
        self.train_dataset = None

        self.test_dataset = next(WikipediaCleanDatasetUnconditional(
            split="test",
            tokenizer=tokenizer,
            max_sequence_len=self.max_sequence_len,
        ).get_data())

    def train_update(self):
        self.train_dataset = next(self.train_dataset_iter)

    def train_dataloader(self):
        self.train_update()
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
