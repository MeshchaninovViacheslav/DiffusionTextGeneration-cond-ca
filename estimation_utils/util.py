import torch
import itertools
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

from utils.util import reduce_tensor, reduce_sum_tensor


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
