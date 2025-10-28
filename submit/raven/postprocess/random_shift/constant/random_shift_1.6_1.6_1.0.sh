#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/random_shift_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/random_shift_%A_%a_%x_%j_%N.err

#SBATCH --job-name random_shift

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72

#SBATCH --time=23:59:59
#SBATCH --partition=standard

# Load required modules
module purge
module load anaconda/3/2021.11

# Activate the conda environment
conda activate avion

cd /u/dduka/work/projects/Thesis/AVION

export PYTHONPATH=.:third_party/decord/python/

python3 -m second_party.postprocess.random_shift \
    --dataset /ptmp/dduka/databases/ego4d/ego4d_train.pkl \
    --output-path /ptmp/dduka/databases/ego4d/rewritten_timestamps/random_shift/ \
    --video-root /ptmp/dduka/databases/ego4d/ego4d/v2/full_scale/ \
    --scale-min 1.6 \
    --scale-max 1.6 \
    --min-duration 1.0 \