#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/lavila_pretrain_sft-ego4d-timelens-8fps-union_vitl_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/lavila_pretrain_sft-ego4d-timelens-8fps-union_vitl_%A_%a_%x_%j_%N.err

#SBATCH --job-name lavila_pretrain_sft-ego4d-timelens-8fps-union_vitl

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72

#SBATCH --time=23:59:59

module purge
module load anaconda/3/2023.03

conda activate avion

export LD_PRELOAD="/raven/u/system/soft/SLE_15/packages/x86_64/gcc/14.1.0/bin/../lib/gcc/x86_64-pc-linux-gnu/14.1.0/../../../../lib64/libstdc++.so.6"
export EK100_TRAIN="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv"
export EK100_VAL="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv"
export EK100_VIDEO_DIR="/ptmp/dduka/databases/EK100/video_320p_15sec/"

export EGTEA_DATA_DIR="/ptmp/dduka/databases/EGTEA/cropped_clips"
export EGTEA_META_DIR="/ptmp/dduka/databases/EGTEA/test_split1.txt"

export CHARADES_DATA_DIR="/ptmp/dduka/databases/charades_ego/CharadesEgo_v1_480"
export CHARADES_META_DIR="/ptmp/dduka/databases/charades_ego/CharadesEgo_v1_480/CharadesEgo/CharadesEgo_v1_test_only1st.csv"

export EGO4D_MCQ_DATA_DIR="/ptmp/dduka/databases/ego4d/video_320px_15sec/"
export EGO4D_MCQ_META_DIR="/ptmp/dduka/databases/ego4d/egovlp2/egomcq.json"

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

RUN_NAME=LAVILA_PRETRAIN_SFT-EGO4D-TIMELENS-8FPS-UNION_VITL14
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
    --model CLIP_VITL14 \
    --train-metadata /ptmp/dduka/databases/ego4d/refined_dataset/sft-ego4d-timelens-8fps-union/refined_lavila_data.pkl \
    --train-metadata-aux /ptmp/dduka/databases/ego4d/ego4d_train.narrator_63690737.return_10.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 28 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-project-name "Main Experiments" \
    --wandb-run-name $RUN_NAME \