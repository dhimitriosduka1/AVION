#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/preprocess_phase2.out
#SBATCH -e /ptmp/dduka/work/logs/avion/preprocess_phase2.err

#SBATCH --job-name preprocess_phase2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=00:59:59

cd /u/dduka/work/projects/Thesis/AVION/

uv run /u/dduka/work/projects/Thesis/AVION/second_party/preprocess/dataset_preprocessing_phase2.py