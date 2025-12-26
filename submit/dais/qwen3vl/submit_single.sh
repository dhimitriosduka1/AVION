#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/single_%A_%a.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/single_%A_%a.err

#SBATCH -J single
#SBATCH --time=01:59:00

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1

#SBATCH --mem=1000000

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate qwen3vl

export PORT=$(shuf -i 10000-65535 -n 1)

cd /u/dduka/project/AVION
export PYTHONPATH=.:third_party/decord/python/

torchrun --nproc_per_node=4 --master_addr localhost --master_port $PORT /u/dduka/project/AVION/second_party/qwen3vl/manual.py \
    --pkl_path /dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_42.pkl \
    --video_root /dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/ \
    --max_new_tokens 256 \
    --output_path /u/dduka/project/AVION/results_single.json \
    --fps 8 \
    --prompt_template v2 \
    --num_workers 4 \
    --only_video_id 0b530687-26d8-4c9d-9771-c758ecd2ecbf \
    --batch_size 2
