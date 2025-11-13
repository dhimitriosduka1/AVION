#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/nouns_and_verbs_gap_%GAP%_%A_%x_%j_%N.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/nouns_and_verbs_gap_%GAP%_%A_%x_%j_%N.err

#SBATCH -J nouns_and_verbs_gap_%GAP%

#SBATCH --time=03:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

module purge
micromamba activate avion_fa2

python3 /u/dduka/project/AVION/second_party/noun_and_verbs/refine_dataset.py --gap ${GAP}