#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/nlq_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/nlq_%A_%a_%x_%j_%N.err

#SBATCH -J nlq
#SBATCH --time=01:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=250000

set -euo pipefail

module purge
micromamba activate avion_fa2

nvidia-smi

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export CUDA_VISIBLE_DEVICES=0

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-unknown}"

cd /u/dduka/project/AVION
export PYTHONPATH=.:third_party/decord/python/

python3 /u/dduka/project/AVION/second_party/nlq_segments/main.py --dataset /dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl