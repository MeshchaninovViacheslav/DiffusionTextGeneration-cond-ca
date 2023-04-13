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

from diffusion_holder_e2e import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL

disable_progress_bar()
set_verbosity_error()


def parse_option(config):
    parser = argparse.ArgumentParser("MMTD")
    if config.ddp:
        parser.add_argument('--local_rank', type=int, required=True)
    args, unparsed = parser.parse_known_args()
    return args


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 5000
    optim.lr = 1e-5
    optim.min_lr = 1e-6
    optim.warmup_lr = 1e-6
    optim.weight_decay = 0

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 1_000
    training.batch_size = 256

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = './checkpoints/end2end_enc_seqlen=32_200_.pth'
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1028
    validation.validation_iters = int(10_000 / validation.batch_size)

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False

    config.model = ml_collections.ConfigDict()
    config.model.ema_rate = 0.9999

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_dif_coef = 0.1
    loss.ce_enc_coef = 0.01
    loss.l0_coef = 0.


    data = config.data = ml_collections.ConfigDict()
    # data.config_path = "/home/vmeshchaninov/DiffusionTextGeneration/latent_diffusion/data/config.json"
    data.max_sequence_len = 32

    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    args = parse_option(config)
    if config.ddp:
        config.local_rank = args.local_rank

    return config


if __name__ == '__main__':
    config = create_config()
    config.checkpoints_prefix = f"e2e-base" \
                                f"ce_dif_coef={config.loss.ce_dif_coef}-" \
                                f"ce_enc_coef={config.loss.ce_enc_coef}-" \
                                f"l0_coef={config.loss.l0_coef}"

    if "base" in config.checkpoints_prefix:
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
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    os.environ['CONFIG_PATH'] = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config, latent_mode="encodings")
    diffusion.train(experiment_name=config.checkpoints_prefix)
