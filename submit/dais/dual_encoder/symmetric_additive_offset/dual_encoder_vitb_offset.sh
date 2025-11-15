#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/sao_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/sao_%A_%a_%x_%j_%N.err

#SBATCH -J sao
#SBATCH --time=23:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

#SBATCH --array=0-6

set -euo pipefail

module purge
micromamba activate avion_fa2

nvidia-smi

# Offsets to sweep
OFFSETS=(1.0 1.5 2.0 2.5 3.0 3.5 4.0)
OFFSET="${OFFSETS[$SLURM_ARRAY_TASK_ID]}"

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Array task ID: $SLURM_ARRAY_TASK_ID  -> OFFSET: $OFFSET"

cd /u/dduka/project/AVION

RUN_BASE=DUAL_ENCODER_ADDITIVE
RUN_NAME=${RUN_BASE}_OFFSET_${OFFSET}_DAIS
EXP_PATH=/dais/fs/scratch/dduka/training_metadata/avion/$RUN_NAME

mkdir -p "$EXP_PATH"

export PYTHONPATH=.:third_party/decord/python/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=4 \
    --nnodes=${SLURM_NNODES:-1} \
    --node_rank=${SLURM_NODEID:-0} \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_backend=c10d \
    scripts/main_lavila_pretrain.py \
    --use-flash-attn \
    --train-metadata /dais/fs/scratch/dduka/databases/ego4d/symmetric_additive_offset/ego4d_train_symmetric_additive_offset_${OFFSET}.pkl \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 512 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir "$EXP_PATH" \
    --wandb-run-name "$RUN_NAME" \
    --workers 32 \
    --decode-threads 4 \
    --prefetch-factor 4 \
    --wandb-group "Master Seminar" \