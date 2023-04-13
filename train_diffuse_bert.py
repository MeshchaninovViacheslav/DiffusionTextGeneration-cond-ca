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
from utils.util import set_seed, _BERT_SMALL, _BERT_BASE_FOR_LARGE_ENC
from diffusion_utils import schedulers

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
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5_000
    optim.lr = 1e-4
    optim.min_lr = 1e-4
    optim.warmup_lr = 1e-6
    optim.weight_decay = 0.

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 1_000_000
    training.checkpoint_freq = 200_000
    training.eval_freq = 50_000
    training.batch_size = 512

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    # refresh.prefix = "./checkpoints/rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3.4-woinputproj_1000000_.pth"
    refresh.prefix = "./checkpoints/rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-new_net-time_t_800000_.pth"
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1024
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 2048
    validation.p_uncond = 0.

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 2000
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False
    sde.scheduler = schedulers.Cosine(sde.beta_min, sde.beta_max) #schedulers.Exponential(a=10)

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.downstream_task = ""  # "qqp"
    model.dataset = "rocstory"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 32

    config.lin_input = True
    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    args = parse_option(config)
    if config.ddp:
        config.local_rank = args.local_rank

    return config


if __name__ == '__main__':
    config = create_config()
    suffix = "cond_launch-v1.0"
    config.checkpoints_prefix = f"{config.model.dataset}-" \
                                f"{config.model.downstream_task if config.model.downstream_task is not None else ''}-" \
                                f"{config.model.embeddings_type}-" \
                                f"prediction={config.model.prediction}-" \
                                f"loss={config.model.loss}-" \
                                f"enc={config.model.enc_type}-" \
                                f"bert={config.model.dif_enc_type}-" \
                                f"kl_cf={config.loss.ce_coef}-" \
                                f"seq_len={config.data.max_sequence_len}-" \
                                f"clipgrad={config.optim.grad_clip_norm}-" \
                                f"lr={config.optim.lr}-" \
                                f"min_lr={config.optim.min_lr}-" \
                                f"lin_input={config.lin_input}-" \
                                f"seed={config.seed}-" \
                                f"wd={config.optim.weight_decay}-" \
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
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.seed
    set_seed(seed)
    os.environ['CONFIG_PATH'] = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train(experiment_name=config.checkpoints_prefix)
