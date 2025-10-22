#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/symmetric_additive_offset_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/symmetric_additive_offset_%A_%a_%x_%j_%N.err

#SBATCH --job-name symmetric_additive_offset

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72

#SBATCH --time=01:59:59
#SBATCH --partition=standard

# Load required modules
module purge
module load anaconda/3/2021.11

# Activate the conda environment
conda activate avion

cd /u/dduka/work/projects/Thesis/AVION

export PYTHONPATH=.:third_party/decord/python/

python3 -m second_party.postprocess.symmetric_additive_offset \
    --dataset /ptmp/dduka/databases/ego4d/ego4d_train.pkl \
    --output-path /ptmp/dduka/databases/ego4d \
    --video-root /ptmp/dduka/databases/ego4d/ego4d/v2/full_scale/ \
    --offset 4.0 \
    --num-workers 70 \