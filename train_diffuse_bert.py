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

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL
from diffusion_utils import schedulers

disable_progress_bar()
set_verbosity_error()

import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")


def parse_option(config):
    parser = argparse.ArgumentParser("MMTD")
    if config.ddp:
        parser.add_argument('--local_rank', type=int, required=True)
    args, unparsed = parser.parse_known_args()
    return args


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 10.
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 1_000_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 100_000
    training.eval_freq = 100_000
    training.batch_size = 512  # * 8

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = "./checkpoints/wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=10.0-lr=0.0002-min_lr=0.0002-seed=0-wd=0.01-batch=512-SD=10-t5-bert-384_100000_.pth"
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1024
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 8192
    validation.p_uncond = 0.

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 200
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False
    sde.coef_d = 10
    sde.scheduler = schedulers.CosineSD(d=sde.coef_d)

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "embeddings"
    model.dif_enc_type = "base"
    model.downstream_task = ""  # "qqp"
    model.dataset = "wikipedia-clean"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.dim = 384
    model.decoder_path = f"decoder-bert-encs-{model.dim}.pth"
    # "decoder-electra-wikipedia-128.pth" #"decoder-roberta_base-wikipedia-128.pth" # "decoder-wikipedia-128.pth"  # "decoder-t5_base-wikipedia-128.pth" "decoder-roberta_base-wikipedia-128.pth"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 96
    data.pos_begin = 0.0
    data.pos_end = 0.67
    data.enc_bert_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-mean.pt"
    data.enc_bert_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-std.pt"
    data.enc_roberta_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-roberta_base-wiki-mean.pt"
    data.enc_roberta_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-roberta_base-wiki-std.pt"
    data.enc_t5_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-mean.pth"
    data.enc_t5_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-std.pth"
    data.enc_electra_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-electra-wiki-mean.pt"
    data.enc_electra_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-electra-wiki-std.pt"
    data.emb_bert_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/embeddings-bert-wiki-mean.pt"
    data.emb_bert_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/embeddings-bert-wiki-std.pt"

    config.finetuning = False
    config.lin_input = True
    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    config.project_name = "bert-encoder-exps"

    return config


if __name__ == '__main__':
    config = create_config()
    suffix = "t5-bert-384"
    config.checkpoints_prefix = f"{config.model.dataset}-" \
                                f"{config.model.downstream_task if config.model.downstream_task is not None else ''}-" \
                                f"prediction={config.model.prediction}-" \
                                f"loss={config.model.loss}-" \
                                f"seq_len={config.data.max_sequence_len}-" \
                                f"cond_seg=[{config.data.pos_begin:0.2f}, {config.data.pos_end:0.2f}]-" \
                                f"clipgrad={config.optim.grad_clip_norm}-" \
                                f"lr={config.optim.lr}-" \
                                f"min_lr={config.optim.min_lr}-" \
                                f"seed={config.seed}-" \
                                f"wd={config.optim.weight_decay}-" \
                                f"batch={config.training.batch_size}-" \
                                f"SD={config.sde.coef_d}-" \
                                f"{suffix}"  # "end2end-enc-base-seqlen32-v.5"  # 'emb_bert_x0_bs=512_lr=2e-4'
    if "base" in config.model.dif_enc_type:
        config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    else:
        config.bert_config = BertConfig(**_BERT_SMALL)

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

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    seed = config.seed
    set_seed(seed)
    os.environ['CONFIG_PATH'] = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train(
        project_name=config.project_name,
        experiment_name=config.checkpoints_prefix
    )
