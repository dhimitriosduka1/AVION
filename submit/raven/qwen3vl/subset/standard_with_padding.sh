#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/scaled_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/scaled_%A_%a_%x_%j_%N.err

#SBATCH --job-name std_padding

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72

#SBATCH --time=15:59:59
#SBATCH --array=0-8

cd /u/dduka/work/projects/Thesis/AVION/

# 1. Configuration
CHUNK_SIZE=50000
TOTAL_ITEMS=362409

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
    --output_file /ptmp/dduka/databases/ego4d/qwen_refinement/standard/output_1_caption_1_padding_362k_subset.jsonl \
    --pkl_path /ptmp/dduka/databases/ego4d/ego4d_train_362k_subset_with_uuid.pkl \
    --video_root /ptmp/dduka/databases/ego4d/video_320px_15sec/ \
    --video_padding 1 \
