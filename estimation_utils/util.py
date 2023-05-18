import torch
import itertools
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

from utils.util import reduce_tensor


def clear_text(text):
    data = []
    for l in text:
        s = l.replace("..", "")
        s = s.replace("[SEP]", "").replace("[CLS]", "")
        s = s.replace("PAD", "").replace("UNK", "").replace("START", "").replace("END", "")
        if not s:
            continue
        if s[0] == ".":
            s = s[1:]
        s = s.strip().lower()
        if s:
            data.append(s)
    return data


def compute_metric(metric_fn, texts):
    num_tokens = 0.0
    metric = 0.0
    T = tqdm(texts)
    for text in T:
        t_metric, t_num = metric_fn(text, reduce="sum")
        if t_metric is None or np.isnan(t_metric):
            continue
        metric += t_metric
        num_tokens += t_num
        T.set_description(f"metric: {metric_fn.name}, {metric / num_tokens:0.4f}")
    return metric / num_tokens


def generate_text(diffusion, num_texts, batch_size):
    generated_texts = []
    while len(generated_texts) < num_texts:
        text = diffusion.generate_text(batch_size=int(min(batch_size, num_texts - len(generated_texts))))[0]
        text = clear_text(text)
        generated_texts += text
    return generated_texts


def generate_text_unconditional(diffusion, num_texts, batch_size):
    generated_texts = []
    while len(generated_texts) < num_texts:
        tmp_batch_size = int(min(batch_size, num_texts - len(generated_texts)))
        dummy_condition = [""] * tmp_batch_size
        dummy_condition = diffusion.tokenizer(
            text=dummy_condition,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=diffusion.config.data.max_sequence_len,
            return_tensors="pt",
        )
        dummy_condition = {"cond": dummy_condition["input_ids"], "cond_mask": dummy_condition["attention_mask"]}
        text = diffusion.generate_text(batch_size=tmp_batch_size, cond=dummy_condition)[0]

        text = clear_text(text)
        generated_texts += text
    return generated_texts


def generate_text_conditional(diffusion, num_texts, batch_size):
    print(batch_size)
    diffusion.config.validation.batch_size = batch_size
    diffusion.set_valid_data_generator()
    loader = iter(diffusion.valid_loader)
    generated_texts = []
    while len(generated_texts) < num_texts:
        tmp_batch_size = int(min(batch_size, num_texts - len(generated_texts)))
        try:
            condition = next(loader)
            condition["cond_ids"] = condition["cond_ids"][:tmp_batch_size]
            condition["cond_mask"] = condition["cond_mask"][:tmp_batch_size]
        except Exception:
            return generated_texts

        condition = {"cond": condition["cond_ids"], "cond_mask": condition["cond_mask"]}
        text = diffusion.generate_text(batch_size=tmp_batch_size, cond=condition)[0]

        joint_text = []
        cond_text = diffusion.tokenizer.batch_decode(condition["cond"], skip_special_tokens=True)
        for i, c_t in enumerate(cond_text):
            joint_text.append(f"{c_t} {text[i]}")

        joint_text = clear_text(joint_text)
        generated_texts += joint_text
    return generated_texts


def estimate_model(diffusion, num_texts, batch_size, metric_bloom_fn, metric_gpt_fn, type_="uncond"):
    if type_ == "uncond":
        texts = generate_text_unconditional(diffusion, num_texts, batch_size)
    elif type_ == "cond":
        texts = generate_text_conditional(diffusion, num_texts, batch_size)
    else:
        texts = generate_text(diffusion, num_texts, batch_size)

    if metric_bloom_fn:
        metric_bloom = compute_metric(metric_bloom_fn, texts)
    else:
        metric_bloom = np.nan
    if metric_gpt_fn:
        metric_gpt = compute_metric(metric_gpt_fn, texts)
    else:
        metric_gpt = np.nan
    return {"Bloom metric": metric_bloom, "GPT2 metric": metric_gpt}, texts


def reduce_metrics(metrics):
    for key, metric in metrics.items():
        # print(f"{dist.get_rank()}, {key}, {metric}", flush=True)
        metric = torch.Tensor([metric]).cuda()
        metric = reduce_tensor(metric).item()
        metrics[key] = metric
        # print(f"{dist.get_rank()}, {key}, {metric}", flush=True)
    return metrics


def gather_texts(texts):
    output = [None for _ in range(dist.get_world_size())]
    gather_objects = texts
    dist.all_gather_object(output, gather_objects)
    gathered_texts = list(itertools.chain(*output))
    return gathered_texts
