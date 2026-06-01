#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/avion/notebook_%j.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/notebook_%j.err
#SBATCH -J debug_run
#SBATCH --time=23:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge

eval "$(micromamba shell hook --shell bash)"
micromamba activate swift

echo "------------------------------------------------"
echo "Job running on node: $SLURMD_NODENAME"
nvidia-smi
echo "------------------------------------------------"

cd /u/dduka/project/AVION

export PYTHONPATH=.:third_party/decord/python/

# Pick a random port to avoid collisions
# (You can also hardcode this if you prefer, e.g., 8888)
PORT=$(shuf -i 8000-9999 -n 1)
NODE=$(hostname)

echo "------------------------------------------------"
echo "   JUPYTER NOTEBOOK RUNNING "
echo "------------------------------------------------"
echo "1. On your local machine, run this SSH tunnel command:"
echo "   ssh -N -L ${PORT}:${NODE}:${PORT} <your_username>@<cluster_login_node>"
echo ""
echo "2. Open this URL in your browser:"
echo "   http://localhost:${PORT}"
echo "------------------------------------------------"

jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0