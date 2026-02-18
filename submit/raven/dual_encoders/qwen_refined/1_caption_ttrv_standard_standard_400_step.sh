#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/lavila_pretrain_baseline_de_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/lavila_pretrain_baseline_de_%A_%a_%x_%j_%N.err
#SBATCH --job-name dual_encoder_pretrain_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72
#SBATCH --time=23:59:59

module purge
module load anaconda/3/2023.03

# OPTIONAL: Load system CUDA if your cluster requires it. 
# Since you need CUDA 12, try enabling this if the LD_LIBRARY_PATH fix alone fails.
module load cuda/12.1

conda activate avion

# --- FIX START: Link Conda Libraries ---
# This forces the system to look for libcudart.so.12 inside your environment first
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# --- FIX END ---

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Debug info
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "LD_LIBRARY_PATH is now: $LD_LIBRARY_PATH"

cd /u/dduka/work/projects/Thesis/AVION

RUN_NAME=DUAL_ENCODER_PRETRAIN_TTRV_STANDARD_STANDARD_400_STEP
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
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 512 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-run-name $RUN_NAME \
    --workers 16 \
    --prefetch-factor 2 \
    --train-metadata /ptmp/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_ttrv_strandard_standard_400_step.pkl \