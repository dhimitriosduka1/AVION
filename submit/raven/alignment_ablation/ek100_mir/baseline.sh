#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/ek100_finetune_baseline_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/ek100_finetune_baseline_%A_%a_%x_%j_%N.err

#SBATCH --job-name ek100_mir_finetune_baseline

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72

#SBATCH --time=23:59:59

module purge
module load anaconda/3/2023.03
module load gcc/14

eval "$(micromamba shell hook --shell bash)"
micromamba activate avion

export LD_PRELOAD="/raven/u/system/soft/SLE_15/packages/x86_64/gcc/14.1.0/bin/../lib/gcc/x86_64-pc-linux-gnu/14.1.0/../../../../lib64/libstdc++.so.6"
export EK100_TRAIN="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv"
export EK100_VAL="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv"
export EK100_VIDEO_DIR="/ptmp/dduka/databases/EK100/video_320p_15sec/"
export RELEVANCY_PATH="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH


echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /u/dduka/work/projects/Thesis/AVION

RUN_NAME=EK100_MIR
EXP_PATH=/ptmp/dduka/work/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/
    
srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=4 scripts/main_lavila_finetune_mir.py \
    --root $EK100_VIDEO_DIR \
    --train-metadata $EK100_TRAIN \
    --val-metadata $EK100_VAL \
    --relevancy-path $RELEVANCY_PATH \
    --video-chunk-length 15 \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 128 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model /u/dduka/work/projects/Thesis/AVION/checkpoints/avion_pretrain_lavila_vitb_best.pt \
    --wandb-run-name $RUN_NAME \
    --wandb \