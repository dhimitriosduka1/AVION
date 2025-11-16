#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/iou_noun_and_verbs_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/iou_noun_and_verbs_%A_%a_%x_%j_%N.err

#SBATCH -J iou_noun_and_verbs
#SBATCH --time=00:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=250000

#SBATCH --array=0-2

set -euo pipefail

module purge
micromamba activate avion_fa2

nvidia-smi

IoUs=(0.005 0.01 0.015 0.02 0.025 0.03)
IoU="${IoUs[$SLURM_ARRAY_TASK_ID]}"

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

python3 /u/dduka/project/AVION/second_party/iou_noun_and_verbs/main.py --min-iou ${IoU}