#!/bin/bash -l
#SBATCH -o /dais/fs/scratch/dduka/logs/avion/sglang_%A_%a.out
#SBATCH -e /dais/fs/scratch/dduka/logs/avion/sglang_%A_%a.err

#SBATCH -J sglang
#SBATCH --time=23:59:00

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1

#SBATCH --mem=1000000

module purge
module load cuda/12.8

eval "$(micromamba shell hook --shell bash)"
micromamba activate sglang

# 2. Define port and model
PORT=30000
MODEL="Qwen/Qwen3-VL-8B-Instruct"

echo "Starting SGLang server on $(hostname) at port $PORT"

python3 -m sglang.launch_server \
  --model-path $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tp 4 \
  --keep-mm-feature-on-device \
  --trust-remote-code \
  --mm-process-config '{"video":{"fps":8}}'