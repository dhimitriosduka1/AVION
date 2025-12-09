#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/dual_encoder_qwen_32b_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/dual_encoder_qwen_32b_%A_%a_%x_%j_%N.err
#SBATCH -J qwen_32b
#SBATCH --time=23:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4

#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge

eval "$(micromamba shell hook --shell bash)"
micromamba activate avion_fa2

echo "------------------------------------------------"
echo "Job running on node: $SLURMD_NODENAME"
echo "GPUs available: $SLURM_GPUS_ON_NODE"
nvidia-smi
echo "------------------------------------------------"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_ID=$SLURM_JOBID

cd /u/dduka/project/AVION

RUN_NAME="DUAL_ENC_HIER_QWEN_2.5_32B"
EXP_PATH="/dais/fs/scratch/dduka/training_metadata/avion/$RUN_NAME"

mkdir -p "$EXP_PATH"

export PYTHONPATH=.:third_party/decord/python/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting training on $MASTER_ADDR:$MASTER_PORT..."

torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_id=$RDZV_ID \
    scripts/main_lavila_pretrain.py \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 512 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir "$EXP_PATH" \
    --wandb-run-name "$RUN_NAME" \
    --train-metadata /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_Qwen2.5-32B-Instruct.pkl