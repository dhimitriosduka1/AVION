#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/preprocess__.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/preprocess__.err

#SBATCH --job-name phase_2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --mem=250000

#SBATCH --time=11:59:59

cd /u/dduka/project/AVION
uv run /u/dduka/project/AVION/second_party/preprocess/dataset_preprocessing_phase2.py