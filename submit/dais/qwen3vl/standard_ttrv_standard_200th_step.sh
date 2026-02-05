#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/ttrv_scaled_standard_%A_%a_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/ttrv_scaled_standard_%A_%a_%x_%j_%N.err

#SBATCH -J ttrv_scaled_standard

#SBATCH --time=15:59:59
#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

#SBATCH --array=0-40%8

cd /u/dduka/project/AVION/

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

module purge
module load gcc/14
module load cuda/12.8

uv run ./second_party/qwen3vl/vllm_refine.py \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --batch_size 512 \
    --tensor_parallel_size 4 \
    --output_file /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/output_1_caption_ttrv_form_standard_checkpoint_200.jsonl \
    --pkl_path /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl \
    --model_path /dais/fs/scratch/dduka/training_metadata/ttrv/checkpoints/TTRL-verl/tag-Qwen/Qwen3-VL-8B-Instruct/TTRL-EGO4D-TAR-STANDARD-SEGMENTS-grpo/global_step_200/actor_merged/