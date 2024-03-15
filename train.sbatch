#!/bin/bash
#SBATCH --job-name="dist2"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=2
#SBATCH --time=2-0:0
#SBATCH --mail-user=egor.chimbulatov@yandex.ru
#SBATCH --mail-type=ALL
#SBATCH --constraint="[type_e]"
#SBATCH --output="distillation"%j.out

# Executable
torchrun --nproc_per_node=1 --master_port=31345 train_diffusion.py
exit

