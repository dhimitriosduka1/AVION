#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/pretrained_checkpoint_evaluation_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/pretrained_checkpoint_evaluation_%A_%a_%x_%j_%N.err

#SBATCH --job-name pretrained_checkpoint_evaluation

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu"
#SBATCH --mem=120000

#SBATCH --time=00:59:59

module purge
module load anaconda/3/2023.03

conda activate avion

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

RUN_NAME=PRETRAINED_CHECKPOINT_EVALUATION_DUAL_ENCODER
EXP_PATH=/ptmp/dduka/work/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/
    
torchrun \
    scripts/main_lavila_pretrain.py \
    --train-metadata /ptmp/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3.pkl \
    --train-metadata-aux /ptmp/dduka/databases/ego4d/ego4d_train.narrator_63690737.return_10.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-run-name $RUN_NAME \
    --resume /u/dduka/work/projects/Thesis/AVION/checkpoints/avion_pretrain_lavila_vitb_best.pt \
    --evaluate \