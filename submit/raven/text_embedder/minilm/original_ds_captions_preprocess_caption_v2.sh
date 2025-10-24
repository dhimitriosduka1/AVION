#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/minilm_embedder_original_ds_captions_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/minilm_embedder_original_ds_captions_%A_%a_%x_%j_%N.err

#SBATCH --job-name minilm_embedder_original_ds_captions

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

python3 -m second_party.text_embedder.models.minillm.main \
    --video-metadata-path /ptmp/dduka/databases/ego4d/unique_captions_preprocess_caption_v2.json \
    --model-name "sentence-transformers/all-MiniLM-L6-v2" \
    --output-path /ptmp/dduka/databases/ego4d/embeddings/ \
    --preprocess-function "preprocess_caption_v2" \
    --flush-frequency 10 \
    --batch-size 4096 \
    --num-workers 8