#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/lavila_10c_std_362K_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/lavila_10c_std_362K_%A_%a_%x_%j_%N.err

#SBATCH -J l_10c_std_362k
#SBATCH --time=15:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500000

module purge
micromamba activate avion_fa2

nvidia-smi

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0,1

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /u/dduka/project/AVION

RUN_NAME=DAIS_LAVILA_10_CAP_STANDARD_362K
EXP_PATH=/dais/fs/scratch/dduka/training_metadata/avion/$RUN_NAME

mkdir -p $EXP_PATH

export PYTHONPATH=.:third_party/decord/python/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
    
srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=2 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/main_lavila_pretrain.py \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 1024 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH \
    --wandb-run-name $RUN_NAME \
    --workers 16 \
    --prefetch-factor 2 \
    --train-metadata /dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train.rephraser.no_punkt_top3_refined_standard_362k_10_caption_vllm.pkl \
    --train-metadata-aux /dais/fs/scratch/dduka/databases/ego4d/ego4d_train.narrator_63690737.return_10.pkl \