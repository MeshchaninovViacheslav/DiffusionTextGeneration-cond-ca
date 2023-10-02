import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
from transformers import BertLMHeadModel, BertTokenizerFast
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import random

from data.dataset import WikipediaDataset
from estimation_utils.metrics import BloomMetricConditional, BloomMetric
from utils.util import dict_to_cuda
from estimation_utils.util import compute_metric

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

dataset = next(WikipediaDataset(
    split="test",
    tokenizer_bert=tokenizer,
    tokenizer_cond=tokenizer,
    tokenizer_gen=tokenizer,
    max_sequence_len=128,
    pos_begin=0.33,
    pos_end=0.67,
).get_data())

batch_size = 128
loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
        )
loader = iter(loader)

metric_bloom_fn = BloomMetric(device="cuda:0")

X = next(loader)

text_cond = tokenizer.batch_decode(
    X["cond_ids"],
    skip_special_tokens=True
)

text_gen = tokenizer.batch_decode(
    X["input_ids"],
    skip_special_tokens=True
)

from estimation_utils.metrics import RobertaMetric
roberta = RobertaMetric(device="cuda:0")


from transformers import pipeline
# gpt2 == gpt2-small
generator_gpt2 = pipeline('text-generation', model='gpt2', device=0)
texts_gpt = generator_gpt2(text_cond, max_new_tokens=64, num_return_sequences=1, return_full_text=False, pad_token_id=50256)
# text_gen = [text[0]["generated_text"] for text in texts_gpt]

texts = [f"{text_cond[i]} {text_gen[i]}" for i in range(batch_size)]

print(compute_metric(metric_bloom_fn, texts=texts))
print("roberta: ", roberta(texts=texts))


texts = [f"{text_cond[i]} {text_gen[i]}" for i in range(batch_size)]
for i, text in enumerate(texts):
    p = 0.1
    text = text.split(" ")
    new_text = []
    for word in text:
        if random.random() < p:
            new_text.append("the")
        else:
            new_text.append(word)

    texts[i] = " ".join(new_text)
print("bloom: ", compute_metric(metric_bloom_fn, texts=texts))
print("roberta: ", roberta(texts=texts))










