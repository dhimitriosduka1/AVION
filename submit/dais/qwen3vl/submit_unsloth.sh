#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/unsloth_%A_%a.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/unsloth_%A_%a.err

#SBATCH -J unsloth
#SBATCH --time=01:59:00

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1

#SBATCH --mem=250000

module purge
module load cuda/12.8

eval "$(micromamba shell hook --shell bash)"
micromamba activate unsloth_env

python3 /u/dduka/project/AVION/second_party/qwen3vl/unsloth_impl.py
