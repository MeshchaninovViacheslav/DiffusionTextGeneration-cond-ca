#!/bin/bash
#SBATCH --job-name="dist1"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=1-0:0
#SBATCH --mail-user=egor.chimbulatov@yandex.ru
#SBATCH --mail-type=ALL
#SBATCH --constraint="[type_e]"

# Executable
git stash
git checkout distillation
torchrun --nproc_per_node=1 --master_port=31346 train_diffusion.py
exit

