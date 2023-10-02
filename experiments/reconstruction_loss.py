import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

import json
import wandb
import torch
import numpy as np
import pandas as pd
import ml_collections
from multiprocessing import Pool
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusion_holder import DiffusionRunner
from transformers import BertConfig, BertTokenizerFast

from data.create_dataset import create_wikipedia_dataset, create_rocstory_dataset, create_glue_unsupervised_dataset
from data.preprocessing import text_preprocessor, unsupervised_preprocessor, supervised_preprocessor
from utils.util import dict_to_tensors, masked_mean, masked_std, set_seed, dict_to_cuda
from utils.ema_model import ExponentialMovingAverage
from diffusion_utils.schedulers import *
from diffusion_utils.schedulers import Cosine, Linear


def create_config():
    config = ml_collections.ConfigDict()

    training = config.training = ml_collections.ConfigDict()
    training.ode_sampling = False
    training.checkpoints_folder = '../checkpoints'
    config.checkpoints_prefix = None

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 100
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False
    sde.scheduler = Cosine(config.sde.beta_min, config.sde.beta_max)

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.prediction = "x_0"
    model.dataset = "rocstory"

    data = config.data = ml_collections.ConfigDict()
    data.config_path = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    data.max_sequence_len = 32

    config.lin_input = False
    config.device = 'cuda:0'
    config.ddp = False
    config.seed = 1
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    return config


def get_model(suffix):
    config = create_config()
    config.sde.scheduler = Cosine(config.sde.beta_min, config.sde.beta_max)
    config.checkpoints_prefix = f"rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=False-seed=0-wd=0.0-sch={suffix}_800000_"
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    seed = config.seed
    set_seed(seed)

    diffusion = DiffusionRunner(config, latent_mode="encodings", eval=True)
    return diffusion


def get_loader():
    max_sequence_len = 32
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset_wiki = create_rocstory_dataset(split="train", tokenizer=tokenizer, max_sequence_len=max_sequence_len)

    set_seed(0)
    batch_size = 1024
    loader = iter(DataLoader(
        dataset_wiki,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
    ))
    return loader


@torch.no_grad()
def compute_restoration_loss(diffusion, loader, timesteps):
    X = next(loader)
    X = dict_to_cuda(X)
    clean_X = diffusion.sampler_emb(X)
    batch_size = clean_X.shape[0]
    mask = X["attention_mask"]

    losses_score_cosine = []
    losses_score_linear = []

    for t in tqdm(timesteps):
        vec_t = t * torch.ones(batch_size, device=diffusion.device)
        marg_forward = diffusion.dynamic.marginal_forward(clean_X, vec_t)
        x_t = marg_forward['x_t']

        scores = diffusion.dynamic.calc_score(diffusion.score_estimator, x_t=x_t, t=vec_t)
        x_0 = scores["x_0"]

        loss_x_0 = diffusion.mse_loss(clean_X, x_0, mask)

        alpha = Cosine(0.1, 20).alpha_std(vec_t)[0][0, 0, 0]
        losses_score_cosine.append((alpha * loss_x_0 / (1 - alpha) ** 2).item())

        alpha = Linear().alpha_std(vec_t)[0][0, 0, 0]
        losses_score_linear.append((alpha * loss_x_0 / (1 - alpha) ** 2).item())

    return losses_score_cosine, losses_score_linear


def wandb_store(step, value, suffix):
    wandb.log({f"{suffix}": value}, step=step)


def main():
    suffix = "cosine"
    experiment_name = f"train=cosine-alpha=cosine-beta=cosine"
    wandb.init(project="bert_diffusion_experiments", name=experiment_name, mode="online")
    diffusion = get_model(suffix)
    loader = get_loader()

    alphas_cosine = []
    eps = 1e-6
    for t in tqdm(torch.linspace(0 + eps, 1 - eps, diffusion.diff_eq_solver.dynamic.N)):
        alpha, _ = cosine(t, diffusion.dynamic.beta_0, diffusion.dynamic.beta_1)
        alphas_cosine.append(alpha.item())

    t_map = []
    for alpha in alphas_cosine:
        if suffix == "cosine":
            t_map.append(cosine_rev(alpha, diffusion.dynamic.beta_0, diffusion.dynamic.beta_1))
        elif suffix == "linear":
            t_map.append(linear_rev(alpha))
        elif suffix == "quadratic":
            t_map.append(quadratic_rev(alpha))

    losses_score_cosine, losses_score_linear = compute_restoration_loss(diffusion, loader, t_map)
    for step, _ in enumerate(t_map):
        wandb_store(step, losses_score_cosine[step], "losses_score")
        #wandb_store(step, losses_score_linear[step], "losses_score")
        wandb_store(step, alphas_cosine[step], "alphas")
        wandb_store(step, t_map[step], "timesteps")


if __name__ == "__main__":
    main()
