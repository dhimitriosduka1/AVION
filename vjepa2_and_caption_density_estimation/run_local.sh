#!/bin/bash -l


module purge
module load cuda/12.8
module load ffmpeg/7.1

micromamba activate vjepa2

nvidia-smi

python3 /u/dduka/project/AVION/vjepa2_and_caption_density_estimation/main.py