#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/verb_aware_semantic_duration_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/verb_aware_semantic_duration_%A_%a_%x_%j_%N.err
#SBATCH -J verb_aware_semantic_duration
#SBATCH --time=03:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4
 
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge

eval "$(micromamba shell hook --shell bash)"
micromamba activate avion_fa2

echo "------------------------------------------------"
echo "Job running on node: $SLURMD_NODENAME"
echo "GPUs available: $SLURM_GPUS_ON_NODE"
nvidia-smi
echo "------------------------------------------------"

# --- Networking Setup for Distributed Data Parallel (DDP) ---
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_ID=$SLURM_JOBID

# Set thread counts to avoid contention between processes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /u/dduka/project/AVION

RUN_NAME="DAIS_VERB_AWARE_SEMANTIC_DURATION"
EXP_PATH="/dais/fs/scratch/dduka/training_metadata/avion/$RUN_NAME"

mkdir -p "$EXP_PATH"

export PYTHONPATH=.:third_party/decord/python/

echo "Starting experiment on $MASTER_ADDR:$MASTER_PORT..."

# --- Detect number of GPUs provided by SLURM ---
# This counts the commas in the GPU list and adds 1 (or defaults to 1)
NUM_GPUS=$(echo $SLURM_GPUS_ON_NODE | awk -F',' '{print NF}')
if [ -z "$NUM_GPUS" ]; then NUM_GPUS=1; fi
echo "Detected $NUM_GPUS GPUs. Launching torchrun..."

# --- Run with Torchrun ---
torchrun \
    --nnodes=4 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /u/dduka/project/AVION/second_party/verb_aware_semantic_duration/main.py \
    --input_path "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl" \
    --output_path "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_verb_aware.pkl" \
    --model_name "facebook/bart-large-mnli" \
    --batch_size 4096 \
    --run_name "$RUN_NAME"