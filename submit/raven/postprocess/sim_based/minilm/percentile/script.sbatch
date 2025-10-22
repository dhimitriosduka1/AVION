#!/bin/bash -l
#SBATCH -o /ptmp/dduka/work/logs/avion/sim_based_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/sim_based_%A_%a_%x_%j_%N.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:59:59
#SBATCH --partition=standard

# Load required modules
module purge
module load anaconda/3/2021.11

# Activate the conda environment
conda activate avion

cd /u/dduka/work/projects/Thesis/AVION
export PYTHONPATH=.:third_party/decord/python/

python3 -m second_party.postprocess.main \
    --dataset /ptmp/dduka/databases/ego4d/ego4d_train.pkl \
    --ego4d-embeddings-path /ptmp/dduka/databases/ego4d/embeddings/sentence-transformers/all-MiniLM-L6-v2/ \
    --lavila-embeddings-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/embeddings/sentence-transformers/all-MiniLM-L6-v2/ \
    --chunk-metadata-root /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7 \
    --embedding-model "sentence-transformers-all-MiniLM-L6-v2" \
    --temperature 0.7 \
    --tau "${TAU}" \
    --embeddings-to-include 10 \
    --output-path /ptmp/dduka/databases/ego4d/similarity_based_shift_timestamps/ \
    --num-workers 70 \
    --mode percentile