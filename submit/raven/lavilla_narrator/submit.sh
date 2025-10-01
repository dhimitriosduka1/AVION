#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/lavila_narrator_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/lavila_narrator_%A_%a_%x_%j_%N.err

#SBATCH --job-name lavila_narrator

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --mem=120000
#SBATCH --constraint="gpu"

#SBATCH --time=00:29:59

module purge
module load anaconda/3/2023.03

conda activate lavilla

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /u/dduka/work/projects/Thesis/AVION

RUN_NAME=LAVILA_NARRATOR
EXP_PATH=/ptmp/dduka/work/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/
    
torchrun \
    --nproc_per_node=1 \
    second_party/lavilla_narrator/main.py \
    --wandb-run-name $RUN_NAME \
    --video-path-root /ptmp/dduka/databases/EK100_320p_15sec_30fps_libx264/video_320p_15sec \