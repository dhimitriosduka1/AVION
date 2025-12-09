#!/bin/bash -l

#SBATCH -o ./qwen7b_ddp_%A_%a.out
#SBATCH -e ./qwen7b_ddp_%A_%a.err
#SBATCH -J qwen7b_ddp

#SBATCH --time=23:00:00

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

#SBATCH --gres=gpu:h200:4
#SBATCH --mem=500G

module purge
micromamba activate avion_fa2

nvidia-smi

export PYTHONPATH=.:third_party/decord/python/
export HF_HOME="/dais/fs/scratch/dduka/hf_cache" 

torchrun --nproc_per_node=4 \
    --master_port=29501 \
    second_party/hierarchical_ds_factory/main.py \
    --input-path /dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl \
    --output-path /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_qwen_pairwise_7b.pkl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 512 \
    --wandb-project "Ego4D-Merging" \
    --wandb-run-name "Qwen-7B-4xGPU-DDP"