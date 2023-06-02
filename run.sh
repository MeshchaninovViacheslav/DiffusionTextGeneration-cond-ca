#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node 6 --master_port 31345 train_diffuse_bert.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 31345 train_diffuse_bert.py
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 31345 estimation.py

##conda activate /home/vmeshchaninov/anaconda3/envs/env
## srun --gpus=1 --gpus-per-node=1  --time=14-0:0 --constraint="type_e" --cpus-per-task=10  --pty /bin/bash