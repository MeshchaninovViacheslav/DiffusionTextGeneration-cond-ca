import os
import sys
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

# disable_progress_bar()
# set_verbosity_error()

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
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 50_000
    training.eval_freq = 100#5_000
    training.batch_size = 512  # * 8

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = "./checkpoints/wikipedia--t5-bert-self_cond_500000_.pth"
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1024
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 8192
    validation.p_uncond = 0.

    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.solver = 'euler'
    dynamic.scheduler = "sd"
    dynamic.N = 200
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.ode_sampling = False
    dynamic.coef_d = 10

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "embeddings"
    model.dif_enc_type = "base"
    model.downstream_task = ""  # "qqp"
    model.dataset = "wikipedia"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.decoder_path = "decoder-wikipedia-128.pth"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 64
    data.pos_begin = 0.0
    data.pos_end = 0.67
    data.enc_bert_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-mean.pt"
    data.enc_bert_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-std.pt"

    data.enc_t5_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-mean.pth"
    data.enc_t5_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-std.pth"

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    config.use_self_cond = True
    config.project_name = "test" #"dtg-exps-1.0"
    config.timesteps = "linear"

    return config


if __name__ == '__main__':
    config = create_config()
    suffix = "test"#f"t5-bert-decoder"
    config.checkpoints_prefix = f"{config.model.dataset}-" \
                                f"{config.model.downstream_task if config.model.downstream_task is not None else ''}-" \
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
