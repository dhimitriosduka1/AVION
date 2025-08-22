#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.err

#SBATCH --job-name avion_env

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=00:59:59

module purge
module load anaconda/3/2021.11
module load cuda/11.8-nvhpcsdk

nvidia-smi
nvcc --version

pixi clean
pixi lock
pixi install
pixi run compile-decord