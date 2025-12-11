#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/qwen_dataloader_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/qwen_dataloader_%A_%a_%x_%j_%N.err
#SBATCH -J qwen_dataloader
#SBATCH --time=11:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4

#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate avion_fa2

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_ID=$SLURM_JOBID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /u/dduka/project/AVION

# --- Config ---
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE=256

# Naming
SHORT_MODEL="Qwen2.5-7B"
RUN_NAME="DAIS_${SHORT_MODEL}_DATALOADER_OPT"

EXP_PATH="/dais/fs/scratch/dduka/training_metadata/avion/$RUN_NAME"
mkdir -p "$EXP_PATH"
export PYTHONPATH=.:third_party/decord/python/

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /u/dduka/project/AVION/second_party/verb_aware_semantic_duration/main.py \
    --input_path "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl" \
    --output_path "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_verb_aware.pkl" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --run_name "$RUN_NAME"