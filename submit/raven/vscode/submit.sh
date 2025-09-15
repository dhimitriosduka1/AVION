#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/%A_%a_%x_%j_%N.err

#SBATCH --job-name debug

#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=23:59:59

module purge
module load avion

sleep infinity