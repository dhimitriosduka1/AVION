#!/bin/bash

JOBS=(
    # afterany:27681703,/ptmp/dduka/work/training_metadata/avion/LAVILA_PRETRAIN_BASELINE/checkpoint_best.pt
    # afterany:27681704,/ptmp/dduka/work/training_metadata/avion/LAVILA_PRETRAIN_ZSH_QWEN_REFINED/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/LAVILA_PRETRAIN_BASELINE_VITL14/checkpoint_best.pt
    # afterany:27681734,/ptmp/dduka/work/training_metadata/avion/LAVILA_PRETRAIN_ZSH_QWEN_REFINED_VITL14/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_EGO4D/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_QWEN_REFINED/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_E4D_VITL14/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_QWEN_REF_VITL14/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_EGO4D_VITL14_/checkpoint_best.pt
    # /ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_QWEN_REFERENCE_VITL14_/checkpoint_best.pt
    afterok:27815993,/ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_LAVITI/checkpoint_best.pt
    afterok:27819393,/ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_LAVITI_VITL/checkpoint_best.pt
    afterok:27818941,/ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_MIDPOINT_EXTENSION/checkpoint_best.pt
    afterok:27819392,/ptmp/dduka/work/training_metadata/avion/DUAL_ENCODER_MIDPOINT_EXTENSION_VITL/checkpoint_best.pt
)

for ENTRY in "${JOBS[@]}"; do
    if [[ "$ENTRY" == *,* ]]; then
        DEP="${ENTRY%%,*}"
        CKPT_DIR="${ENTRY#*,}"
        DEP_DIRECTIVE="#SBATCH --dependency=${DEP}"
    else
        DEP=""
        CKPT_DIR="$ENTRY"
        DEP_DIRECTIVE=""
    fi

    if [[ "$CKPT_DIR" == *.pt ]]; then
        CKPT_PATH="$CKPT_DIR"
        CKPT_DIR=$(dirname "$CKPT_PATH")
    else
        CKPT_PATH="${CKPT_DIR}/checkpoint_best.pt"
    fi

    if [ -z "$DEP" ] && [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: ${CKPT_PATH} not found, skipping."
        continue
    fi

    DIR_NAME=$(basename "$CKPT_DIR")
    RUN_NAME="EK100_MIR_${DIR_NAME^^}"
    JOB_NAME="ek100_mir_${DIR_NAME}"

    if [[ "${DIR_NAME,,}" == *vitl* ]]; then
        NODES=4
        BATCH_SIZE=32
    else
        NODES=2
        BATCH_SIZE=64
    fi

    echo "Submitting job: ${JOB_NAME}"
    echo "Checkpoint:     ${CKPT_PATH}"
    echo "WandB run name: ${RUN_NAME}"
    echo "Nodes: ${NODES}  |  Batch size: ${BATCH_SIZE}"
    [ -n "$DEP" ] && echo "Dependency:     ${DEP}"

    sbatch <<EOF
#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/${JOB_NAME}_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/avion/${JOB_NAME}_%j_%N.err

#SBATCH --job-name ${JOB_NAME}

#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=72

#SBATCH --time=23:59:59
${DEP_DIRECTIVE}

module purge
module load anaconda/3/2023.03
module load gcc/14

eval "\$(micromamba shell hook --shell bash)"
micromamba activate avion

export LD_PRELOAD="/raven/u/system/soft/SLE_15/packages/x86_64/gcc/14.1.0/bin/../lib/gcc/x86_64-pc-linux-gnu/14.1.0/../../../../lib64/libstdc++.so.6"
export EK100_TRAIN="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv"
export EK100_VAL="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv"
export EK100_VIDEO_DIR="/ptmp/dduka/databases/EK100/video_320p_15sec/"
export RELEVANCY_PATH="/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"

export EGTEA_DATA_DIR="/ptmp/dduka/databases/EGTEA/cropped_clips"
export EGTEA_META_DIR="/ptmp/dduka/databases/EGTEA/test_split1.txt"

export CHARADES_DATA_DIR="/ptmp/dduka/databases/charades_ego/CharadesEgo_v1_480"
export CHARADES_META_DIR="/ptmp/dduka/databases/charades_ego/CharadesEgo_v1_480/CharadesEgo/CharadesEgo_v1_test_only1st.csv"

export EGO4D_MCQ_DATA_DIR="/ptmp/dduka/databases/ego4d/video_320px_15sec/"
export EGO4D_MCQ_META_DIR="/ptmp/dduka/databases/ego4d/egovlp2/egomcq.json"

export MASTER_PORT=\$((12000 + \$RANDOM % 20000))
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/lib64:\$LD_LIBRARY_PATH

echo "Job running on nodes: \$SLURM_JOB_NODELIST"
echo "Total nodes: \$SLURM_NNODES"
echo "GPUs per node: \$SLURM_GPUS_ON_NODE"

cd /u/dduka/work/projects/Thesis/AVION

EXP_PATH="/ptmp/dduka/work/training_metadata/avion/${RUN_NAME}"
mkdir -p \$EXP_PATH

export PYTHONPATH=.:third_party/decord/python/

srun --cpu_bind=v --accel-bind=gn torchrun \\
    --nproc_per_node=4 \\
    --nnodes=\$SLURM_NNODES \\
    --node_rank=\$SLURM_NODEID \\
    --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \\
    --rdzv_backend=c10d \\
    scripts/main_lavila_finetune_mir.py \\
    --root \$EK100_VIDEO_DIR \\
    --train-metadata \$EK100_TRAIN \\
    --val-metadata \$EK100_VAL \\
    --relevancy-path \$RELEVANCY_PATH \\
    --video-chunk-length 15 \\
    --use-flash-attn \\
    --grad-checkpointing \\
    --use-fast-conv1 \\
    --batch-size ${BATCH_SIZE} \\
    --fused-decode-crop \\
    --use-multi-epochs-loader \\
    --pretrain-model ${CKPT_PATH} \\
    --wandb \\
    --wandb-project "Main Experiments" \\
    --wandb-run-name ${RUN_NAME} \\
    --output-dir \$EXP_PATH
EOF

    echo "------------------------------------------------------"
done
