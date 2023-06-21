import torch
import itertools
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

from utils.util import reduce_tensor, reduce_sum_tensor


def clear_text(text):
    data = []
    for l in text:
        s = l.replace("..", "")
        s = s.replace("[SEP]", "").replace("[CLS]", "")
        s = s.replace("PAD", "").replace("UNK", "").replace("START", "").replace("END", "")
        if not s:
            data.append("")
            continue
        if s[0] == ".":
            s = s[1:]
        s = s.strip().lower()
        data.append(s)
    return data

@torch.no_grad()
def compute_metric(metric_fn, cond_texts=None, gen_texts=None, texts=None):
    num_tokens = 0.0
    metric = 0.0
    length = len(cond_texts) if texts is None else len(texts)
    T = tqdm(range(length))
    for i in T:
        if texts is None:
            t_metric, t_num = metric_fn(cond_text=cond_texts[i], gen_text=gen_texts[i], reduce="sum")
        else:
            t_metric, t_num = metric_fn(text=texts[i], reduce="sum")
        if t_metric is None or np.isnan(t_metric) or t_num == 0:
            continue
        metric += t_metric
        num_tokens += t_num
        T.set_description(f"metric: {metric_fn.name}, {metric / num_tokens:0.4f}")
    return metric / num_tokens

@torch.no_grad()
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

@torch.no_grad()
def generate_text_conditional(diffusion, num_texts, batch_size):
    print(batch_size)
    diffusion.config.validation.batch_size = batch_size
    diffusion.set_valid_data_generator()
    loader = iter(diffusion.valid_loader)

    # diffusion.config.training.batch_size_per_gpu = batch_size
    # diffusion.set_train_data_generator()
    # loader = iter(diffusion.train_loader)

    cond_texts = []
    gen_texts = []
    gt_texts = []
    joint_texts = []
    while len(gen_texts) < num_texts:
        tmp_batch_size = int(min(batch_size, num_texts - len(gen_texts)))
        try:
            condition = next(loader)
            condition["cond_ids"] = condition["cond_ids"][:tmp_batch_size]
            condition["cond_mask"] = condition["cond_mask"][:tmp_batch_size]
            condition["input_ids"] = condition["input_ids"][:tmp_batch_size]
            condition["input_mask"] = condition["input_mask"][:tmp_batch_size]
        except Exception:
            return gen_texts

        gen_text = diffusion.generate_text(
            batch_size=tmp_batch_size,
            cond={"cond": condition["cond_ids"], "cond_mask": condition["cond_mask"]},
            attention_mask=None, #condition["input_mask"],
        )[0]

        cond_text = diffusion.tokenizer_cond.batch_decode(condition["cond_ids"], skip_special_tokens=True)
        gt_text = diffusion.tokenizer_gen.batch_decode(condition["input_ids"], skip_special_tokens=True)

        joint_text = []
        for i, c_t in enumerate(cond_text):
            joint_text.append(f"{c_t} {gen_text[i]}")

        joint_text = clear_text(joint_text)
        gen_text = clear_text(gen_text)
        cond_text = clear_text(cond_text)

        cond_texts += cond_text
        gen_texts += gen_text
        gt_texts += gt_text
        joint_texts += joint_text

    return joint_texts, cond_texts, gen_texts, gt_texts

@torch.no_grad()
def estimate_model(diffusion, num_texts, batch_size, metric_bloom_fn, metric_roberta_fn):
    joint_texts, cond_texts, gen_texts, gt_texts = generate_text_conditional(diffusion, num_texts, batch_size)

    if metric_bloom_fn:
        metric_bloom = compute_metric(metric_bloom_fn, cond_texts, gen_texts)
    else:
        metric_bloom = np.nan

    if metric_roberta_fn:
        metric_roberta = metric_roberta_fn(texts=gen_texts)
    else:
        metric_roberta = np.nan
    return {"Bloom metric": metric_bloom,
            "Roberta metric": metric_roberta}, joint_texts, cond_texts, gen_texts, gt_texts


def reduce_metrics(metrics):
    for key, metric in metrics.items():
        # print(f"{dist.get_rank()}, {key}, {metric}", flush=True)
        metric = torch.Tensor([metric]).cuda()
        metric = reduce_tensor(metric).item()
        metrics[key] = metric
        # print(f"{dist.get_rank()}, {key}, {metric}", flush=True)
    return metrics


def reduce_sum_metrics(metrics):
    for key, metric in metrics.items():
        metric = torch.Tensor([metric]).cuda()
        metric = reduce_sum_tensor(metric).item()
        metrics[key] = metric
    return metrics


def gather_texts(texts):
    output = [None for _ in range(dist.get_world_size())]
    gather_objects = texts
    dist.all_gather_object(output, gather_objects)
    gathered_texts = list(itertools.chain(*output))
    return gathered_texts
