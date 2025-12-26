#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/ego4d_%A_%a.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/ego4d_%A_%a.err

#SBATCH -J original
#SBATCH --time=01:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1

#SBATCH --mem=1000000

# Create a job array
#SBATCH --array=0-999%15

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate qwen3vl

echo "Job index: $SLURM_ARRAY_TASK_ID"

# --- FIX START ---
# Option 1: Load into array (Use parentheses)
video_ids=($(cat /u/dduka/project/AVION/second_party/target_video_ids.txt))
video_id=${video_ids[$SLURM_ARRAY_TASK_ID]}

# Check if video_id is empty (safety check)
if [ -z "$video_id" ]; then
    echo "Error: No video_id found for index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Processing video_id: $video_id"
# --- FIX END ---

export PORT=$(shuf -i 10000-65535 -n 1)

cd /u/dduka/project/AVION
export PYTHONPATH=.:third_party/decord/python/

torchrun --nproc_per_node=4 --master_addr localhost --master_port $PORT /u/dduka/project/AVION/second_party/qwen3vl/manual.py \
    --pkl_path /dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl \
    --video_root /dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/ \
    --max_new_tokens 256 \
    --only_video_id $video_id \
    --output_path /u/dduka/project/AVION/second_party/qwen3vl/output_original/results_${video_id}.json \
    --fps 8