import os
from tqdm import tqdm
import json
from datasets import load_dataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from multiprocessing.pool import Pool


disable_progress_bar()
set_verbosity_error()

base_path = "/home/gbartosh/nlp_models/data/c4/en"
files = os.listdir(base_path)

def func(file):
    if "lock" in file:
        return
    try:
        d = load_dataset(path=base_path, data_files=file)
    except Exception:
        print(f"{file} is broken", flush=True)


with Pool() as pool:
    pool.map(func, files)


