#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/preprocess.out
#SBATCH -e /ptmp/dduka/work/logs/avion/preprocess.err

#SBATCH --job-name preprocess_ds_2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:2
#SBATCH --mem=240000

#SBATCH --time=09:59:59

cd /u/dduka/work/projects/Thesis/AVION/

# 4. Run the Python Script
uv run /u/dduka/work/projects/Thesis/AVION/second_party/preprocess/dataset_preprocessing.py