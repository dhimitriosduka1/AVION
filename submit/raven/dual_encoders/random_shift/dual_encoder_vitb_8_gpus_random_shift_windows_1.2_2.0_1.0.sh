#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/lavila_pretrain_baseline_de_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/lavila_pretrain_baseline_de_%A_%a_%x_%j_%N.err

#SBATCH --job-name dual_encoder_pretrain_baseline_8_gpus

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72

#SBATCH --time=15:59:59
#SBATCH --wait-all-nodes=1

module purge
module load anaconda/3/2023.03

conda activate avion

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /u/dduka/work/projects/Thesis/AVION

RUN_NAME=DUAL_ENCODER_PRETRAIN_BASELINE_256_RANDOM_SHIFT_WIND-1.2-2.0-1.0
EXP_PATH=/ptmp/dduka/work/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/
    
srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/main_lavila_pretrain.py \
    --train-metadata /ptmp/dduka/databases/ego4d/random_shift_timestamps/ego4d_train_random_shift_1.2_2.0_1.0.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-run-name $RUN_NAME \
    --wandb-group "Random Timestamp Shift Runs"