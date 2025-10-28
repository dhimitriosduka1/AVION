#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.err

#SBATCH --job-name clip_embedder_pe_core_bigg-14-448_original_ds_captions

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu"
#SBATCH --mem=120000

#SBATCH --time=03:59:59

module purge
module load anaconda/3/2023.03

conda activate open_clip

nvidia-smi

cd /u/dduka/work/projects/Thesis/AVION
export PYTHONPATH=/u/dduka/work/projects/Thesis/AVION:$PYTHONPATH

python3 -m second_party.text_embedder.models.clip.main \
    --video-metadata-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/unique_captions_preprocess_caption_v2.json \
    --output-path /ptmp/dduka/databases/ego4d/embeddings/lavila_narrator/0.7/preprocess_caption_v2/ \
    --preprocess-function preprocess_caption_v2 \
    --model-name PE-Core-bigG-14-448 \
    --pretrained meta \
    --flush-frequency 100 \
    --batch-size 4096 \
    --num-workers 8 \