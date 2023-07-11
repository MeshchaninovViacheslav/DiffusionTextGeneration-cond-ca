import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import json
import torch
from metrics import BloomMetricConditional, RobertaMetric
from estimation_utils.util import clear_text, compute_metric
from estimation_utils.diversity_metrics import NGramStats


def read_file(text_file):
    texts = json.load(open(text_file, "r"))
    return texts


@torch.no_grad()
def estimate_file(text_file):
    print(text_file)
    texts = read_file(text_file)
    cond_texts = [d["CONDITION"] for d in texts]
    gen_texts = [d["GEN"] for d in texts]
    gt_texts = [d["GT"] for d in texts]

    metric_bloom_fn = BloomMetricConditional(device=f"cuda:0")
    metric_roberta_fn = RobertaMetric(device=f"cuda:0")
    print(f"Metrics are loaded")

    metric_bloom = compute_metric(metric_bloom_fn, cond_texts=cond_texts, gen_texts=gt_texts)
    metric_gpt = metric_roberta_fn(texts=gt_texts)[0]

    print(f"Bloom metric: {metric_bloom:0.5f}")
    print(f"Roberta score: {metric_gpt:0.5f}")

    metric_div = NGramStats()
    metric_div.compute(gt_texts)
    print(metric_div)


if __name__ == "__main__":
    for file in [
        "wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-t5-bert-womask_800000_-num_texts=8192-scale=3.0.json",
        #"gpt2-500_000.json"
    ]:
        #text_file = f"../lm_training/generated_texts/{file}"
        text_file = f"../generated_texts/{file}"
        estimate_file(text_file)
