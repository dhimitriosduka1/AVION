#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.err

#SBATCH --job-name lavila_embedder

#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=03:59:59

module purge
module load anaconda/3/2023.03

conda activate lavila

nvidia-smi

python3 -m second_party.text_embedder.models.lavila.main \
    --video-metadata-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/unique_captions.json \
    --output-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/embeddings/lavila \
    --batch-size 256 \
    --num-workers 8