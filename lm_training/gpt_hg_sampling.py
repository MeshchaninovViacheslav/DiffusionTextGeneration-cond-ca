import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import json
from transformers import BertLMHeadModel, BertTokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from data.dataset import WikipediaDatasetDDP
from estimation_utils.metrics import BloomMetricConditional, RobertaMetric
from utils.util import dict_to_cuda, set_seed
from estimation_utils.util import compute_metric
from estimation_utils.diversity_metrics import NGramStats


def get_dataloader(tokenizer):
    dataset = next(WikipediaDatasetDDP(
        split="valid",
        tokenizer_bert=tokenizer,
        tokenizer_cond=tokenizer,
        tokenizer_gen=tokenizer,
        max_sequence_len=64,
        pos_begin=0.,
        pos_end=0.67,
    ).get_data())

    batch_size = 128
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )
    return loader


def generate(text_cond, max_length, num_beams=1):
    texts_gpt = generator_gpt2(
        text_cond,
        max_new_tokens=max_length,
        num_return_sequences=1,
        return_full_text=False,
        pad_token_id=50256,
        num_beams=num_beams,
    )

    return [l[0]["generated_text"] for l in texts_gpt]


def generate_texts(total_num_texts, max_length):
    num_texts = 0
    total_gen_texts = []
    total_cond_texts = []
    total_gt_texts = []

    for X in tqdm(loader):
        text_cond = tokenizer.batch_decode(
            X["cond_ids"],
            skip_special_tokens=True,
        )
        text_gt = tokenizer.batch_decode(
            X["input_ids"],
            skip_special_tokens=True,
        )

        total_gen_texts += generate(text_cond, max_length=max_length)
        total_cond_texts += text_cond
        total_gt_texts += text_gt

        num_texts += len(text_cond)
        if num_texts >= total_num_texts:
            break
    return total_cond_texts, total_gen_texts, total_gt_texts


if __name__ == "__main__":
    set_seed(0)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    loader = get_dataloader(tokenizer)

    # gpt2 == gpt2-small
    generator_gpt2 = pipeline('text-generation', model='gpt2', device=0)
    total_num_texts = 8196
    max_length = 50

    cond_texts, gen_texts, gt_texts = generate_texts(total_num_texts, max_length)

    text_list = []
    for i in range(len(cond_texts)):
        text_list.append(
            {
                "CONDITION": cond_texts[i],
                "GEN": gen_texts[i],
                "GT": gt_texts[i]
            }
        )

    texts_path = "../generated_texts"
    model_name = "gpt-hg"
    prefix = f"{total_num_texts}"
    file_name = f"{texts_path}/{model_name}-{prefix}.json"
    json.dump(text_list, open(file_name, "w"), indent=4)
    print(file_name)

    # Metrics
    metric_bloom_fn = BloomMetricConditional(device=f"cuda:0")
    metric_roberta_fn = RobertaMetric(device=f"cuda:0")

    metric_bloom = compute_metric(metric_bloom_fn, cond_texts, gen_texts)
    metric_roberta = metric_roberta_fn(texts=gen_texts)[0]
    print(f"Bloom metric: {metric_bloom:0.5f}")
    print(f"Roberta metric: {metric_roberta:0.5f}")

    metric_div = NGramStats()
    metric_div.compute(gen_texts)
    print(metric_div)

