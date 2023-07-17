import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import lightning as L
from torch.utils.data import DataLoader

from data.dataset_clean_wiki import WikipediaCleanDatasetUnconditional
from data.dataset_wiki import WikipediaDatasetUnconditional
from data.dataset import SST2Dataset


class WikiDataModule(L.LightningDataModule):
    def __init__(self, tokenizer, config, collate_fn=None):
        super(WikiDataModule, self).__init__()

        self.max_sequence_len = config.data.max_sequence_len
        self.batch_size = config.training.batch_size
        self.collate_fn = collate_fn
        self.num_workers = config.data.num_workers
        self.bert_recon_dataset = config.data.bert_recon_dataset

        self.train_dataset_iter = WikipediaDatasetUnconditional(
            split="train",
            tokenizer=tokenizer,
            max_sequence_len=self.max_sequence_len,
        ).get_data()
        self.train_dataset = None

        self.valid_dataset = next(WikipediaCleanDatasetUnconditional(
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
            shuffle=True,
        )

    def val_dataloader(self):
        if self.bert_recon_dataset:
            loaders = {
                "mask": DataLoader(
                    dataset=self.valid_dataset,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn,
                    num_workers=self.num_workers,
                ),
                "clean": DataLoader(
                    dataset=self.valid_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )
            }
            return loaders

        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

class SSTDataModule(L.LightningDataModule):
    def __init__(self, tokenizer, config, collate_fn=None):
        super(SSTDataModule, self).__init__()

        self.max_sequence_len = config.data.max_sequence_len
        self.batch_size = config.training.batch_size
        self.collate_fn = collate_fn
        self.num_workers = config.data.num_workers
        self.bert_recon_dataset = config.data.bert_recon_dataset

        self.train_dataset = next(SST2Dataset(
            split="train",
            tokenizer_bert=tokenizer,
            tokenizer_cond=tokenizer,
            tokenizer_gen=tokenizer,
            max_sequence_len=self.max_sequence_len,
        ).get_data())

        self.valid_dataset = next(SST2Dataset(
            split="test",
            tokenizer_bert=tokenizer,
            tokenizer_cond=tokenizer,
            tokenizer_gen=tokenizer,
            max_sequence_len=self.max_sequence_len,
        ).get_data())

        self.test_dataset = next(SST2Dataset(
            split="test",
            tokenizer_bert=tokenizer,
            tokenizer_cond=tokenizer,
            tokenizer_gen=tokenizer,
            max_sequence_len=self.max_sequence_len,
        ).get_data())

    def train_update(self):
        self.train_dataset = self.train_dataset

    def train_dataloader(self):
        self.train_update()
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        loaders = {
            "clean": DataLoader(
                dataset=self.valid_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        }
        return loaders

    def predict_dataloader(self):
        loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader


