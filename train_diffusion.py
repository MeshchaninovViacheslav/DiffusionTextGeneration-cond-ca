import torch.distributed as dist

from diffusion_holder import DiffusionRunner
from utils import set_seed, setup_ddp
from create_config import create_config


if __name__ == '__main__':
    config = create_config()
    setup_ddp()

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    if dist.get_rank() == 0:
        print(config)

    seed = config.seed
    set_seed(seed)

    diffusion = DiffusionRunner(config)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train()
