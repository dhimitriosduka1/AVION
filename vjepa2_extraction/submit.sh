#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/vjepa2_extraction.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/vjepa2_extraction.err

#SBATCH -J vjepa2_extraction
#SBATCH --time=23:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge
module load cuda/12.8
module load ffmpeg/7.1

micromamba activate vjepa2

nvidia-smi

python3 /u/dduka/project/AVION/vjepa2_extraction/main.py