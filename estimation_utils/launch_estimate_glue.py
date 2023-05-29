import os
import torch
import psutil
import datasets
import argparse
import ml_collections
import torch.distributed as dist
from datasets import disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from transformers import BertConfig

datasets.config.IN_MEMORY_MAX_SIZE = psutil.virtual_memory().available

import sys
sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from diffusion_holder import DiffusionRunner
from utils.util import set_seed
from diffusion_utils import schedulers

disable_progress_bar()
set_verbosity_error()


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 0
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 2e-4
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 400_000
    training.finetuning_iters = 10_000
    training.training_iters = training.training_iters + training.finetuning_iters
    training.checkpoint_freq = 1_000
    training.eval_freq = 1_000
    training.batch_size = 512

    training.ode_sampling = False
    training.checkpoints_folder = '../checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = True
    refresh.prefix = "./checkpoints/wikipedia--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-cond-bert_400000_.pth"
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1024
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 2048
    validation.p_uncond = 0.

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 100
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False
    sde.scheduler = schedulers.CosineSD(d=10)

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.downstream_task = "sst2"  # "qqp"
    model.dataset = "glue"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 64

    config.lin_input = True
    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    config.project_name = "bert-conditional-exps"

    return config


if __name__ == '__main__':
    config = create_config()
    suffix = "glue-sst2"
    config.checkpoints_prefix = "glue-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-glue-sst2_405000_"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.seed
    set_seed(seed)
    os.environ['CONFIG_PATH'] = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=True)

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    accuracy = diffusion.estimate_finetuning()
    if dist.get_rank() == 0:
        print(f"accuracy: {accuracy}")
