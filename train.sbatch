#!/bin/bash
#SBATCH -A proj_1525
#SBATCH --job-name="dist2"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=1-0:0
#SBATCH --mail-user=egor.chimbulatov@yandex.ru
#SBATCH --mail-type=ALL
#SBATCH --constraint="[type_e]"
#SBATCH --output="sbatch_out/learning_teacher_no_boundary"%j.out

# Executable
torchrun --nproc_per_node=1 --master_port=31347 train_diffusion.py
exit

