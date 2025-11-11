#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/4_dual_encoder_pretrain_baseline_dais_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/4_dual_encoder_pretrain_baseline_dais_%A_%a_%x_%j_%N.err

#SBATCH -J 4_dual_encoder_pretrain_baseline_dais
#SBATCH --time=23:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge
micromamba activate avion_fa2

nvidia-smi

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /u/dduka/project/AVION

RUN_NAME=DUAL_ENCODER_PRETRAIN_BASELINE_DAIS_4GPU
EXP_PATH=/dais/fs/scratch/dduka/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/
    
srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/main_lavila_pretrain.py \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 512 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-run-name $RUN_NAME \