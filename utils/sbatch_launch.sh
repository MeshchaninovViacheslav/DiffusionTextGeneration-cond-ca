#!/bin/bash
#SBATCH --job-name="t5-model-ln"
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gpus-per-node=4
#SBATCH --time=3-1:0
#SBATCH --mail-user=meshchaninov01@mail.ru
#SBATCH --mail-type=ALL
#SBATCH --constraint="[type_e]"

# Executable
torchrun --nproc_per_node=4 --master_port=31345  train_diffuse_bert.py
