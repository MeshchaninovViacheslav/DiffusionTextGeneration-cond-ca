import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import json
from evaluate import load

def compute_mauve(file_name):
    mauve = load('mauve')

    data = json.load(open(file_name, "r"))
    predictions = [d["GEN"] for d in data]
    references = [d["GT"] for d in data]

    device_id = 0
    mauve_results = mauve.compute(predictions=predictions, references=references, device_id=device_id).mauve
    return mauve_results

dir = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/generated_texts/" 

files_list = [
    # "gpt2-500_000.json",
    # "wikipedia--t5-bert-self_cond_last_-num_texts=8196-scale=0.0.json",
    # "wikipedia--t5-bert-self_cond_last_-num_texts=8196-scale=2.0.json",
    #"wikipedia--self_cond_time_shift-my_bert_last_-num_texts=8196-scale=0.5.json"
    "wikipedia--t5-bert-initial_last_-num_texts=8196-scale=0.0.json"
]
results = dict()
for file_name in files_list:
    try:
        mauve = compute_mauve(os.path.join(dir, file_name))
        results[file_name] = mauve
    except Exception:
        print(file_name, "exception")

metrics_file = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/metrics/mauve_results-1.json" 
with open(metrics_file, "w") as file:
    json.dump(results, file)




# 0.85
# 0.854