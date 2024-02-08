import os
import sys
import torch
import psutil
import datasets
import torch.distributed as dist

datasets.config.IN_MEMORY_MAX_SIZE = psutil.virtual_memory().available

from diffusion_holder import DiffusionRunner
from utils.util import set_seed
from create_config import create_config


sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")


if __name__ == '__main__':
    config = create_config()
    suffix = f"{config.data.max_sequence_len}"
    config.training.checkpoints_prefix = f"cond-{config.model.dataset}-{config.model.cond_encoder_name_hash}-{config.model.encoder_name_hash}-{suffix}"  

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

    diffusion = DiffusionRunner(config)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train(
        project_name=config.project_name,
        experiment_name=config.checkpoints_prefix
    )
