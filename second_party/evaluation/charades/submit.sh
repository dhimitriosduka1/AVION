#!/bin/bash -l
#SBATCH -o /ptmp/dduka/work/logs/avion/charades_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/charades_%A_%a_%x_%j_%N.err

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

python3 -m second_party.evaluation.charades.dataset