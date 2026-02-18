#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/std_scl_ttrv_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/std_scl_ttrv_%A_%a_%x_%j_%N.err

#SBATCH --job-name std_scl_ttrv

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72

#SBATCH --time=14:59:59

#SBATCH --array=0-40%20

cd /u/dduka/work/projects/Thesis/AVION

CHUNK_SIZE=100000
TOTAL_ITEMS=4012544

START_IDX=$(( $SLURM_ARRAY_TASK_ID * $CHUNK_SIZE ))
END_IDX=$(( $START_IDX + $CHUNK_SIZE ))

if [ $END_IDX -gt $TOTAL_ITEMS ]; then
    END_IDX=$TOTAL_ITEMS
fi

echo "================================================="
echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Processing:   $START_IDX to $END_IDX"
echo "Batch Size:   $CHUNK_SIZE"
echo "================================================="

uv run ./second_party/qwen3vl/vllm_refine.py \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --batch_size 2048 \
    --tensor_parallel_size 4 \
    --output_file /ptmp/dduka/databases/ego4d/qwen_refinement/standard/output_1_caption_ttrv_form_scaled_checkpoint_400.jsonl \
    --pkl_path /ptmp/dduka/databases/ego4d/ego4d_train_with_uuid.pkl \
    --model_path /ptmp/dduka/work/training_metadata/ttrv/scale/actor_merged/ \
    --video_root /ptmp/dduka/databases/ego4d/video_320px_15sec/ \