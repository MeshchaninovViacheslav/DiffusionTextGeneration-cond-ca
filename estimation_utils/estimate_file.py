import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

import torch
from metrics import BloomMetric, GPTNEOMetric
from estimation_utils.util import clear_text, compute_metric


def read_file(text_file):
    with open(text_file, "r") as file:
        texts = file.readlines()
    return clear_text(texts)


@torch.no_grad()
def estimate_file(text_file):
    print(text_file)
    texts = read_file(text_file)

    metric_bloom_fn = BloomMetric(device=f"cuda:0")
    metric_gpt_fn = GPTNEOMetric(device=f"cuda:0")
    print(f"Metrics are loaded")

    metric_bloom = compute_metric(metric_bloom_fn, texts)
    metric_gpt = compute_metric(metric_gpt_fn, texts)

    print(f"Bloom metric: {metric_bloom:0.5f}")
    print(f"GPTNEO metric: {metric_gpt:0.5f}")


if __name__ == "__main__":
    # for file in os.listdir("../generated_texts"):
    #     text_file = f"../generated_texts/{file}"
    #     estimate_file(text_file)

    for file in ["conditioned_cosine.txt", "conditioned_cosine_lin.txt"]:
        text_file = f"../generated_texts/{file}"
        estimate_file(text_file)

    # f"/home/vmeshchaninov/DiffusionTextGeneration/generated_texts/encodings-x0-bs=512-wopad-mask-base-roc-lr=sch-ce_500000_.txt"
    # "/home/vmeshchaninov/DiffusionTextGeneration/generated_texts/encodings-x0-bs=512-wopad-mask-base-roc-lr=sch-ce-seq=64_400000_.txt"
    # f"/home/vmeshchaninov/DiffusionTextGeneration/generated_texts/encodings-x0-bs=512-wopad-mask-base-roc-lr=sch-ce_500000_.txt"
