#! /bin/bash
#SBATCH -A proj_1525
#SBATCH --job-name="wiki"
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-0:0
#SBATCH --mail-user=egor.chimbulatov@yandex.ru
#SBATCH --mail-type=ALL
#SBATCH --output="sbatch_out/wiki_basic"%j.out
#SBATCH --constraint="type_e"

python stats.py
torchrun --nproc_per_node=8 train_diffusion.py
exit