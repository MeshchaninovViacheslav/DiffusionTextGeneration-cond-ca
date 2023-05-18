import torch
import random
import numpy as np
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def set_seed(seed: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True


def dict_to_cuda(d):
    for key in d:
        d[key] = d[key].cuda(non_blocking=True)
    return d


def dict_to_tensor_cuda(d):
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key not in d:
            continue
        d[key] = torch.Tensor(d[key]).cuda(non_blocking=True)
    return d


def dict_to_tensors(d):
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        d[key] = torch.tensor(d[key])
    return d


def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def masked_mean(tensor, mask):
    return torch.sum(tensor * mask[:, :, None], dim=[0, 1]) / torch.sum(mask)


def masked_std(tensor, mask):
    mean = masked_mean(tensor, mask)
    return torch.sqrt(torch.sum(tensor ** 2 * mask[:, :, None], dim=[0, 1]) / torch.sum(mask) - mean ** 2)


def parse_checkpoint_name(checkpoint_name):
    items = checkpoint_name.split("-")
    params = dict()
    for item in items:
        key, value = item.split("=")
        params[key] = value
    return params


def make_mask_wo_SEP_CLS(mask):
    mask = deepcopy(mask)
    mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
    mask[:, 0] = 0
    return mask


def get_ravel_weights(model):
    ww = []
    for par in model.parameters():
        ww.append(par.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


def get_ravel_grad(model):
    ww = []
    for par in model.parameters():
        ww.append(par.grad.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


_BERT_SMALL = {
    "hidden_size": 768,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 8,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 4,
    "intermediate_size": 2048,
    "attention_probs_dropout_prob": 0.1
}

_BERT_BASE = {
    "hidden_size": 768,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 12,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "model_type": "bert",
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
}

_BERT_BASE_FOR_LARGE_ENC = {
    "hidden_size": 1024,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 12,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "model_type": "bert",
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
}

_BERT_LARGE = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
    "type_vocab_size": 2,
    "vocab_size": 30522
}
