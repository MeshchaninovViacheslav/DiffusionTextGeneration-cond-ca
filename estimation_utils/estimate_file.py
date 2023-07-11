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

    metric_bloom = compute_metric(metric_bloom_fn, cond_texts=cond_texts, gen_texts=gen_texts)
    metric_gpt = metric_roberta_fn(texts=gen_texts)[0]

    print(f"Bloom metric: {metric_bloom:0.5f}")
    print(f"Roberta score: {metric_gpt:0.5f}")

    metric_div = NGramStats()
    metric_div.compute(gen_texts)
    print(metric_div)


if __name__ == "__main__":
    for file in ["gpt2-500_000.json"]:
        text_file = f"../lm_training/generated_texts/{file}"
        estimate_file(text_file)
