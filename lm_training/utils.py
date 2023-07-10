import torch

def calc_model_grads_norm(model: torch.nn.Module, p: float = 2):
    grads = []
    for par in model.parameters():
        if par.requires_grad and par.grad is not None:
            grads += [torch.sum(par.grad ** p)]
    return torch.pow(sum(grads), 1. / p)

