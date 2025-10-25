#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/compute_unique_captions_0.7_process_captions_v2_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/compute_unique_captions_0.7_process_captions_v2_%A_%a_%x_%j_%N.err

#SBATCH --job-name compute_unique_captions_0.7_process_captions_v2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72

#SBATCH --time=11:59:59
#SBATCH --partition=standard

# Load required modules
module purge
module load anaconda/3/2021.11

# Activate the conda environment
conda activate open_clip

pip install wordfreq

cd /u/dduka/work/projects/Thesis/AVION

export PYTHONPATH=.:third_party/decord/python/

python3 -m second_party.preprocess.resolve_unique_captions \
    --root-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/ \
    --preprocess-function preprocess_caption_v2 \