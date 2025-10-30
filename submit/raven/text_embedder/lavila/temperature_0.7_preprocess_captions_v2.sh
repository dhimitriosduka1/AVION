#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/lavila_embedder_original_ds_preprocess_caption_v2_%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/lavila_embedder_original_ds_preprocess_caption_v2_%A_%a_%x_%j_%N.err

#SBATCH --job-name lavila_embedder_original_ds_preprocess_caption_v2

#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:1
#SBATCH --mem=120000

#SBATCH --time=03:59:59

module purge
module load anaconda/3/2023.03

conda activate avion

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU visibility and CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

cd /u/dduka/work/projects/Thesis/AVION

export PYTHONPATH=.:third_party/decord/python/

python3 -m second_party.text_embedder.models.lavila.main \
    --resume /u/dduka/work/projects/Thesis/AVION/checkpoints/avion_pretrain_lavila_vitb_best.pt \
    --video-metadata-path /ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_0.7/unique_captions_preprocess_caption_v2.json \
    --model-name "lavila" \
    --output-path /ptmp/dduka/databases/ego4d/embeddings/lavila_narrator/0.7/preprocess_caption_v2 \
    --preprocess-function "preprocess_caption_v2" \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --freeze-temperature \
    --flush-frequency 10 \
    --batch-size 4096 \
    --num-workers 8