#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/standard_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/standard_%A_%a_%x_%j_%N.err

#SBATCH -J standard_vllm

#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

# --- JOB ARRAY CONFIGURATION ---
# Total Items: 4,012,544
# Chunk Size:  100,000
# Total Tasks: 4,012,544 / 100,000 = 40.12 -> 41 tasks.
# Array Indices: 0 to 40
#SBATCH --array=0-40%20

cd /u/dduka/project/AVION/

# 1. Configuration
CHUNK_SIZE=100000
TOTAL_ITEMS=4012544

# 2. Calculate Start and End Indices
START_IDX=$(( $SLURM_ARRAY_TASK_ID * $CHUNK_SIZE ))
END_IDX=$(( $START_IDX + $CHUNK_SIZE ))

# 3. Safety Cap for the final job (Task 40)
# This prevents the script from trying to process beyond the dataset length.
if [ $END_IDX -gt $TOTAL_ITEMS ]; then
    END_IDX=$TOTAL_ITEMS
fi

echo "================================================="
echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Processing:   $START_IDX to $END_IDX"
echo "Batch Size:   $CHUNK_SIZE"
echo "================================================="

# 4. Run the Python Script
uv run ./second_party/qwen3vl/vllm_refine.py \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --batch_size 512 \
    --tensor_parallel_size 4 \
    --output_file /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/output_1_caption.jsonl \
    --pkl_path /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl