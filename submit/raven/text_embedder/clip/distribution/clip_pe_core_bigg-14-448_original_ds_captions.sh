#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/clip_embedder_pe_core_bigg-14-448_dist_original_ds_captions_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/clip_embedder_pe_core_bigg-14-448_dist_original_ds_captions_%A_%a_%x_%j_%N.err

#SBATCH --job-name clip_embedder_pe_core_bigg-14-448_dist_original_ds_captions

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
    --source-run-name "PE-Core-bigG-14-448_meta_ego4d" \
    --embeddings-path /ptmp/dduka/databases/ego4d/embeddings/PE-Core-bigG-14-448_meta/ \
    --method "subset" \
    --n-pairs 1000000 \
    --bins 200 \
    --smooth-sigma-bins 1.2 \
    --seed 42 \