#!/bin/bash
#SBATCH -A proj_1525
#SBATCH --job-name="dist4"
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=4
#SBATCH --time=2-0:0
#SBATCH --mail-user=egor.chimbulatov@yandex.ru
#SBATCH --mail-type=ALL
#SBATCH --constraint="[type_e]"
#SBATCH --output="sbatch_out/frozen_teacher_bc_no_clip_disc_time"%j.out

# Executable
torchrun --nproc_per_node=4 --master_port=31345 train_diffusion.py
exit