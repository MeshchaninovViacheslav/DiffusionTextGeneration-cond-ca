#!/bin/bash
#SBATCH --job-name="estimation"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=0-1:0
#SBATCH --mail-user=meshchaninov01@mail.ru
#SBATCH --mail-type=ALL
#SBATCH --constraint="[type_e]"

# Executable
torchrun --nproc_per_node=1 --master_port=31345  conditional_estimation_for_conditional_model.py
