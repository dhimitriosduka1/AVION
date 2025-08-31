#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.err

#SBATCH --job-name lavila_pretrain_baseline

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=03:59:59

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: 1"
echo "Total GPUs: $((SLURM_NNODES * 1))"

cd /u/dduka/work/projects/AVION

RUN_NAME=LAVILA_PRETRAIN_BASELINE_DEBUG_ONE_GPU
EXP_PATH=/ptmp/dduka/work/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

PYTHONPATH=.:third_party/decord/python/ pixi run torchrun \
    --nproc_per_node=1 \
    scripts/main_lavila_pretrain.py \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --wandb-run-name $RUN_NAME \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt