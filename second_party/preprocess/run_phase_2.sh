#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/preprocess.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/preprocess.err

#SBATCH --job-name preprocess_ds_2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=00:29:59

cd /u/dduka/project/AVION
uv run /u/dduka/project/AVION/second_party/preprocess/dataset_preprocessing_phase2.py