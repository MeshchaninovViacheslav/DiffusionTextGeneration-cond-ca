import sys

from data.dataset import DatasetDDP
from estimation_utils.evaluation import compute_perplexity, compute_mauve
from estimation_utils.diversity_metrics import NGramStats
from create_config import create_config

from transformers import AutoTokenizer
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
split = "train"
max_context_len = 64
dataset_path = "home/vmeshchaninov/nlp_models/data/ag_news"

config = create_config()

dataset = next(DatasetDDP(
    config=config, 
    split=split
).get_data())

def crop(texts, max_length):
    tokens = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        add_special_tokens=True,
        return_tensors="np",
    )["input_ids"]
    tokens = tokens[:, :max_length]
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return texts


size = 10000
n_samples = 5


result_dict = {
    "perplexity": {},
    "div": {},
    "mauve": {},
}

for i in range(n_samples):
    indexes = np.random.default_rng().permutation(len(dataset))[:size]
    dt_sample_1 = dataset[indexes]

    indexes = np.random.default_rng().permutation(len(dataset))[:size]
    dt_sample_2 = dataset[indexes]
    
    for l in [64, 128, 256]:
        target_list_1 = crop(dt_sample_1['text'], max_length=l)
        target_list_2 = crop(dt_sample_2['text'], max_length=l)

        
        print("perplexity...")
        perplexity = compute_perplexity(
            target_list_1, 
        )
        if l not in result_dict["perplexity"]:
            result_dict["perplexity"][l] = []
        result_dict["perplexity"][l].append(perplexity)
        json.dump(result_dict, open("result_reference_metrics.json", "w"))
        
        
        print("NGramStats...")
        metric_div = NGramStats()
        metric_div.compute(target_list_1)

        if l not in result_dict["div"]:
            result_dict["div"][l] = []
        result_dict["div"][l].append(metric_div.results["diversity"])
        json.dump(result_dict, open("result_reference_metrics.json", "w"))


        print("Mauve...")        
        mauve = compute_mauve(target_list_1, target_list_2)
        if l not in result_dict["mauve"]:
            result_dict["mauve"][l] = []
        result_dict["mauve"][l].append(mauve)
        json.dump(result_dict, open("result_reference_metrics.json", "w"))
         