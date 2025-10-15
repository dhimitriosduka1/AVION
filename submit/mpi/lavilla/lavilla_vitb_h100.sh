#!/bin/bash -l

#SBATCH -o /BS/dduka/work/logs/avion/lavila_pretrain_baseline_de_%A_%a_%x_%j_%N.out
#SBATCH -e /BS/dduka/work/logs/avion/lavila_pretrain_baseline_de_%A_%a_%x_%j_%N.err

#SBATCH --job-name lavila_pretrain_baseline

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition gpu24
#SBATCH --gres=gpu:4

#SBATCH --time=23:59:59
#SBATCH --array=1-3%1

micromamba activate avion_h100

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /BS/dduka/work/projects/AVION/

RUN_NAME=LAVILA_PRETRAIN_BASELINE_512_4_GPUS_MPI
EXP_PATH=/BS/dduka/work/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/
    
srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/main_lavila_pretrain.py \
    --train-metadata /scratch/inf0/user/dduka/ego4d/ego4d_train.rephraser.no_punkt_top3.pkl \
    --train-metadata-aux /scratch/inf0/user/dduka/ego4d/ego4d_train.narrator_63690737.return_10.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 512 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-run-name $RUN_NAME \