import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

import json
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

from NPEET.npeet import entropy_estimators as ee


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 10.
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-6
    optim.weight_decay = 0

    training = config.training = ml_collections.ConfigDict()
    training.ode_sampling = False
    training.checkpoints_folder = '../checkpoints'
    training.training_iters = 1_000_000
    training.checkpoint_freq = 200_000
    training.eval_freq = 10_000
    training.batch_size = 512

    config.checkpoints_prefix = ""
    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 2000
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.prediction = "eps"
    model.dataset = "rocstory"

    data = config.data = ml_collections.ConfigDict()
    data.config_path = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    data.max_sequence_len = 32

    config.lin_input = False
    config.device = 'cuda:0'
    config.ddp = False
    config.seed = 1
    config.bert_config = None

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = "../checkpoints/wikipedia--encodings-prediction=eps-loss=L_eps-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=10.0-lr=0.0002-min_lr=0.0002-lin_input=False-seed=1-gradient_exploding_17600_.pth"
    return config


def get_model():
    config = create_config()
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    seed = config.seed
    set_seed(seed)

    diffusion = DiffusionRunner(config, latent_mode="encodings", eval=False)
    return diffusion


def get_loader():
    max_sequence_len = 16
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset_wiki = create_rocstory_dataset(split="train", tokenizer=tokenizer, max_sequence_len=max_sequence_len)

    set_seed(0)
    batch_size = 1024 // 3
    loader = iter(DataLoader(
        dataset_wiki,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
    ))
    return loader


def main():
    global pool_func

    diffusion = get_model()
    loader = get_loader()

    batch = next(loader)
    X = dict_to_cuda(batch)
    clean_X = diffusion.sampler_emb(X)


    results_mi = dict()
    batch_size = clean_X.size(0)
    n = 40
    num_features = 768

    x_bank = dict()
    y_bank = dict()
    t_linspace = [t.item() for t in torch.linspace(0, 1, n)]
    with torch.no_grad():
        for t in t_linspace:
            vec_t = torch.ones(batch_size).to(clean_X.device) * t
            marg_forward = diffusion.sde.marginal_forward(clean_X, vec_t)
            x_t, noise = marg_forward['x_t'], marg_forward['noise']
            pred_embeddings = diffusion.denormalize(x_t)
            tokens = diffusion.decoder(pred_embeddings).argmax(dim=-1)
            X_t = deepcopy(batch)
            X_t["input_ids"] = tokens
            E_D_x_t = diffusion.sampler_emb(X_t)
            print(torch.mean((tokens == batch["input_ids"]) * 1.), torch.norm(E_D_x_t - clean_X))

            x = clean_X[1:-1].detach().cpu().numpy().reshape(-1, 768)
            x = x[:, :num_features]
            y = E_D_x_t[1:-1].detach().cpu().numpy().reshape(-1, 768)
            y = y[:, :num_features]

            x_bank[t] = x
            y_bank[t] = y

    print("The banks are ready")

    def pool_func(t, x, y):
        return ee.mi(x, y, base=np.e, k=3)

    # for t in tqdm(torch.linspace(0, 1, n)):
    #     pool_func(t)
    st_time = time()
    args = [(t, x_bank[t], y_bank[t]) for t in t_linspace]

    with Pool() as pool:
        res = pool.starmap(pool_func, args)

    for i, t in enumerate(t_linspace):
        results_mi[t] = res[i]

    with open(f"results_mi_dec_{num_features}f.json", "w") as file:
        json.dump(results_mi, file)

    print(f"Time = {(time() - st_time) / 3600:0.3f} hours")

if __name__ == "__main__":
    main()
