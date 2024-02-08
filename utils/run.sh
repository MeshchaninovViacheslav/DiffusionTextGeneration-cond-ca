#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node 6 --master_port 31345 train_diffuse_bert.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 31345 train_diffuse_bert.py
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 31345 estimation.py

##conda activate /home/vmeshchaninov/anaconda3/envs/env
## srun --gpus=4 --gpus-per-node=4  --time=2-0:0 --constraint="type_e" --cpus-per-task=4  --pty /bin/bash

srun --gpus=4 --gpus-per-node=4  --time=14-0:0 --constraint="type_e" --cpus-per-task=15  --pty /bin/bash

srun --gpus=2 --gpus-per-node=2  --time=2-0:0 --constraint="type_e" --cpus-per-task=2  --pty /bin/bash

srun --gpus=1 --gpus-per-node=1  --time=1-0:0 --cpus-per-task=5  --pty /bin/bash

srun  --time=1-0:0 --cpus-per-task=40  --pty /bin/bash


torchrun --nproc_per_node=1 --master_port=31345  train_diffuse_bert.py

python utils/kill_wandb.py





