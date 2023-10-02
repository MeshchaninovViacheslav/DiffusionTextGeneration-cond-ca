import psutil
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from time import time
from transformers import BertTokenizerFast

from dataset import C4BatchDataset

from create_dataset import create_unsupervised_dataset


def print_dict(x):
    for key in x:
        print(f"{key}: {x[key]}")


def print_used_RAM():
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

def set_seeds(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
dataset = iter(C4BatchDataset(
            num_batch_data_files=32,
            tokenizer=tokenizer,
            max_sequence_len=32,
        ))

dt = next(dataset)


print_used_RAM()
