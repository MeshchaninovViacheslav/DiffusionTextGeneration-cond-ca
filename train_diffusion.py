import os
import torch
import torch.distributed as dist

from diffusion_holder import DiffusionRunner
from utils.util import set_seed
from create_config import create_config


if __name__ == '__main__':
    config = create_config()
    suffix = ""
    if config.model.conditional_encoder_train:
        suffix += "-cond_train"
    suffix += f"-lr={config.optim.lr}"
    suffix += f"-min_lr={config.optim.min_lr}"
    config.training.checkpoints_prefix = f"{config.data.dataset_name}-{config.model.conditional_encoder_name_hash}-{config.model.encoder_name_hash}-batch={config.training.batch_size}{suffix}"  

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
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train(
        project_name=config.project_name,
        experiment_name=config.training.checkpoints_prefix
    )
