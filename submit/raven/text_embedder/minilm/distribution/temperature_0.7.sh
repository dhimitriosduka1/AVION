#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/minilm_embedder_dist_original_ds_captions_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/minilm_embedder_dist_original_ds_captions_%A_%a_%x_%j_%N.err

#SBATCH --job-name minilm_embedder_dist_original_ds_captions

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu"
#SBATCH --mem=120000

#SBATCH --time=00:59:59

module purge
module load anaconda/3/2023.03

conda activate open_clip

nvidia-smi

cd /u/dduka/work/projects/Thesis/AVION
export PYTHONPATH=/u/dduka/work/projects/Thesis/AVION:$PYTHONPATH

python3 -m second_party.postprocess.resolve_embeddings_sim_dist \
    --source-run-name "sentence-transformers/all-MiniLM-L6-v2/temperature_0.7" \
    --embeddings-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/embeddings/sentence-transformers/all-MiniLM-L6-v2/ \
    --method "subset" \
    --n-pairs 1000000 \
    --bins 200 \
    --smooth-sigma-bins 1.2 \
    --seed 42 \