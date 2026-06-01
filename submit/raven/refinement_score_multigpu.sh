#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/avion/segment_ranking_multigpu_%j.out
#SBATCH -e /ptmp/dduka/work/logs/avion/segment_ranking_multigpu_%j.err
#SBATCH --job-name=segment_rank_mgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu"
#SBATCH --cpus-per-task=48
#SBATCH --time=23:59:59
#SBATCH --mem=160G

set -euo pipefail

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

stage() {
    log "===== $* ====="
}

trap 'status=$?; log "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

stage "Environment setup"
module purge
module load anaconda/3/2023.03
module load gcc/14

eval "$(micromamba shell hook --shell bash)"
micromamba activate avion

export LD_PRELOAD="/raven/u/system/soft/SLE_15/packages/x86_64/gcc/14.1.0/bin/../lib/gcc/x86_64-pc-linux-gnu/14.1.0/../../../../lib64/libstdc++.so.6"

cd /u/dduka/work/projects/Thesis/AVION
export PYTHONPATH=.:third_party/decord/python/

log "hostname=$(hostname)"
log "slurm_job_id=${SLURM_JOB_ID:-manual}"
log "slurm_job_name=${SLURM_JOB_NAME:-manual}"
log "cwd=$(pwd)"
log "python=$(which python)"
python --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

INPUT_MANIFEST="${INPUT_MANIFEST:-/ptmp/dduka/databases/ego4d/refined_dataset/baseline_qwen3vl/refined_data.pkl}"
LAVILA_CHECKPOINT="${LAVILA_CHECKPOINT:-/ptmp/dduka/work/training_metadata/avion/qwen3vl_full_refined_lavila_baseline/}"
EGOVLP_CHECKPOINT="${EGOVLP_CHECKPOINT:-/ptmp/dduka/work/training_metadata/avion/qwen3vl_refined_full_baseline/}"
VIDEO_ROOT="${VIDEO_ROOT:-/ptmp/dduka/databases/ego4d/video_320px_15sec/}"
VIDEO_DURATIONS="${VIDEO_DURATIONS:-/ptmp/dduka/databases/ego4d/video_lengths.json}"
UUID_FILTER_CSV="${UUID_FILTER_CSV-/u/dduka/work/projects/Thesis/thesis/test.csv}"
UUID_COLUMN="${UUID_COLUMN:-uuid}"
CONFIG="${CONFIG:-configs/refinement/segment_ranking.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/dduka/work/segment_ranking/poc_test_csv_multigpu}"
TEST_ANNOTATIONS="${TEST_ANNOTATIONS-${UUID_FILTER_CSV}}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.1,0.3,0.5,0.7,0.9}"
ANNOTATION_IOU_OUTPUT="${ANNOTATION_IOU_OUTPUT:-$OUTPUT_DIR/test_annotation_iou.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-fp16}"
LOG_EVERY="${LOG_EVERY:-1000}"
RANK_SHARDS="${RANK_SHARDS:-8}"

if [[ -z "$LAVILA_CHECKPOINT" && -z "$EGOVLP_CHECKPOINT" ]]; then
    echo "Set at least one of LAVILA_CHECKPOINT or EGOVLP_CHECKPOINT." >&2
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_IDS=(0 1 2 3)
fi
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
if (( NUM_GPUS < 1 )); then
    echo "NUM_GPUS must be >= 1" >&2
    exit 1
fi
if (( NUM_GPUS > ${#GPU_IDS[@]} )); then
    echo "NUM_GPUS=$NUM_GPUS but only ${#GPU_IDS[@]} CUDA_VISIBLE_DEVICES entries are available: ${GPU_IDS[*]}" >&2
    exit 1
fi
GPU_IDS=("${GPU_IDS[@]:0:$NUM_GPUS}")

mkdir -p "$OUTPUT_DIR"

CANDIDATES="$OUTPUT_DIR/candidates.jsonl"
SCORED_CANDIDATES="$OUTPUT_DIR/scored_candidates.jsonl"
REFINED_MANIFEST="$OUTPUT_DIR/refined_manifest.jsonl"
CANDIDATE_RANKINGS="$OUTPUT_DIR/candidate_rankings.jsonl"
METRICS="$OUTPUT_DIR/metrics.json"
SCORE_SHARD_DIR="$OUTPUT_DIR/score_shards_${SLURM_JOB_ID:-$$}"

stage "Configuration"
log "INPUT_MANIFEST=$INPUT_MANIFEST"
log "LAVILA_CHECKPOINT=$LAVILA_CHECKPOINT"
log "EGOVLP_CHECKPOINT=$EGOVLP_CHECKPOINT"
log "VIDEO_ROOT=$VIDEO_ROOT"
log "VIDEO_DURATIONS=$VIDEO_DURATIONS"
log "UUID_FILTER_CSV=${UUID_FILTER_CSV:-<disabled>}"
log "UUID_COLUMN=$UUID_COLUMN"
log "CONFIG=$CONFIG"
log "OUTPUT_DIR=$OUTPUT_DIR"
log "TEST_ANNOTATIONS=${TEST_ANNOTATIONS:-<disabled>}"
log "IOU_THRESHOLDS=$IOU_THRESHOLDS"
log "ANNOTATION_IOU_OUTPUT=$ANNOTATION_IOU_OUTPUT"
log "BATCH_SIZE=$BATCH_SIZE"
log "DEVICE=$DEVICE"
log "INFERENCE_PRECISION=$INFERENCE_PRECISION"
log "LOG_EVERY=$LOG_EVERY"
log "NUM_GPUS=$NUM_GPUS"
log "GPU_IDS=${GPU_IDS[*]}"
log "RANK_SHARDS=$RANK_SHARDS"

stage "Build candidates"
BUILD_ARGS=(
    --input "$INPUT_MANIFEST"
    --output "$CANDIDATES"
    --config "$CONFIG"
    --video-durations "$VIDEO_DURATIONS"
    --log-every "$LOG_EVERY"
)
if [[ -n "$UUID_FILTER_CSV" ]]; then
    BUILD_ARGS+=(--uuid-filter-csv "$UUID_FILTER_CSV" --uuid-column "$UUID_COLUMN")
fi
python -u tools/refinement/build_candidates.py "${BUILD_ARGS[@]}"
log "candidate_file=$CANDIDATES lines=$(wc -l < "$CANDIDATES")"

stage "Score candidates on multiple GPUs"
rm -rf "$SCORE_SHARD_DIR"
mkdir -p "$SCORE_SHARD_DIR"
split -n "l/$NUM_GPUS" -d --additional-suffix=.jsonl \
    "$CANDIDATES" \
    "$SCORE_SHARD_DIR/candidates.part_"

mapfile -t CANDIDATE_SHARDS < <(find "$SCORE_SHARD_DIR" -maxdepth 1 -name 'candidates.part_*.jsonl' | sort)
log "created ${#CANDIDATE_SHARDS[@]} candidate shards"

PIDS=()
LOGS=()
SCORED_PARTS=()
for shard_idx in "${!CANDIDATE_SHARDS[@]}"; do
    gpu_id="${GPU_IDS[$shard_idx]}"
    shard="${CANDIDATE_SHARDS[$shard_idx]}"
    scored_part="$SCORE_SHARD_DIR/scored_candidates.part_$(printf '%04d' "$shard_idx").jsonl"
    shard_log="$SCORE_SHARD_DIR/score_shard_$(printf '%04d' "$shard_idx").log"
    SCORED_PARTS+=("$scored_part")
    LOGS+=("$shard_log")

    SCORE_ARGS=(
        --candidates "$shard"
        --output "$scored_part"
        --config "$CONFIG"
        --video-root "$VIDEO_ROOT"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --inference-precision "$INFERENCE_PRECISION"
        --log-every "$LOG_EVERY"
    )
    if [[ -n "$LAVILA_CHECKPOINT" ]]; then
        SCORE_ARGS+=(--lavila-checkpoint "$LAVILA_CHECKPOINT")
    fi
    if [[ -n "$EGOVLP_CHECKPOINT" ]]; then
        SCORE_ARGS+=(--egovlp-checkpoint "$EGOVLP_CHECKPOINT")
    fi

    log "starting score shard $shard_idx gpu=$gpu_id candidates=$shard log=$shard_log"
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u tools/refinement/score_segments.py "${SCORE_ARGS[@]}" \
        > "$shard_log" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for shard_idx in "${!PIDS[@]}"; do
    pid="${PIDS[$shard_idx]}"
    if wait "$pid"; then
        log "score shard $shard_idx finished"
    else
        FAILED=1
        log "score shard $shard_idx failed; last log lines:"
        tail -n 120 "${LOGS[$shard_idx]}" || true
    fi
done
if (( FAILED != 0 )); then
    exit 1
fi

log "concatenating scored candidate shards"
cat "${SCORED_PARTS[@]}" > "$SCORED_CANDIDATES"
log "scored_candidate_file=$SCORED_CANDIDATES lines=$(wc -l < "$SCORED_CANDIDATES")"

stage "Rank segments"
if (( RANK_SHARDS <= 1 )); then
    python -u tools/refinement/rank_segments.py \
        --scores "$SCORED_CANDIDATES" \
        --input "$INPUT_MANIFEST" \
        --output "$REFINED_MANIFEST" \
        --candidate-output "$CANDIDATE_RANKINGS" \
        --config "$CONFIG" \
        --log-every "$LOG_EVERY"
else
    RANK_SHARD_DIR="$OUTPUT_DIR/rank_shards_${SLURM_JOB_ID:-$$}"
    rm -rf "$RANK_SHARD_DIR"
    mkdir -p "$RANK_SHARD_DIR"
    split -n "l/$RANK_SHARDS" -d --additional-suffix=.jsonl \
        "$SCORED_CANDIDATES" \
        "$RANK_SHARD_DIR/scored_candidates.part_"

    mapfile -t SCORE_SHARDS < <(find "$RANK_SHARD_DIR" -maxdepth 1 -name 'scored_candidates.part_*.jsonl' | sort)
    log "created ${#SCORE_SHARDS[@]} ranking shards"

    PIDS=()
    LOGS=()
    REFINED_PARTS=()
    RANKING_PARTS=()
    for shard_idx in "${!SCORE_SHARDS[@]}"; do
        shard="${SCORE_SHARDS[$shard_idx]}"
        refined_part="$RANK_SHARD_DIR/refined_manifest.part_$(printf '%04d' "$shard_idx").jsonl"
        ranking_part="$RANK_SHARD_DIR/candidate_rankings.part_$(printf '%04d' "$shard_idx").jsonl"
        shard_log="$RANK_SHARD_DIR/rank_shard_$(printf '%04d' "$shard_idx").log"
        REFINED_PARTS+=("$refined_part")
        RANKING_PARTS+=("$ranking_part")
        LOGS+=("$shard_log")
        log "starting rank shard $shard_idx scores=$shard log=$shard_log"
        python -u tools/refinement/rank_segments.py \
            --scores "$shard" \
            --input "$INPUT_MANIFEST" \
            --output "$refined_part" \
            --candidate-output "$ranking_part" \
            --config "$CONFIG" \
            --log-every "$LOG_EVERY" \
            > "$shard_log" 2>&1 &
        PIDS+=("$!")
    done

    FAILED=0
    for shard_idx in "${!PIDS[@]}"; do
        pid="${PIDS[$shard_idx]}"
        if wait "$pid"; then
            log "rank shard $shard_idx finished"
        else
            FAILED=1
            log "rank shard $shard_idx failed; last log lines:"
            tail -n 80 "${LOGS[$shard_idx]}" || true
        fi
    done
    if (( FAILED != 0 )); then
        exit 1
    fi

    log "concatenating ranked shard outputs"
    cat "${REFINED_PARTS[@]}" > "$REFINED_MANIFEST"
    cat "${RANKING_PARTS[@]}" > "$CANDIDATE_RANKINGS"
fi
log "refined_manifest=$REFINED_MANIFEST lines=$(wc -l < "$REFINED_MANIFEST")"
log "candidate_rankings=$CANDIDATE_RANKINGS lines=$(wc -l < "$CANDIDATE_RANKINGS")"

stage "Evaluate diagnostics"
EVAL_ARGS=(
    --input "$REFINED_MANIFEST"
    --qwen-input "$INPUT_MANIFEST"
    --scores "$SCORED_CANDIDATES"
    --output "$METRICS"
)
if [[ -n "$TEST_ANNOTATIONS" ]]; then
    EVAL_ARGS+=(
        --test-annotations "$TEST_ANNOTATIONS"
        --annotation-output "$ANNOTATION_IOU_OUTPUT"
        --iou-thresholds "$IOU_THRESHOLDS"
    )
fi
python -u tools/refinement/evaluate_ranking.py "${EVAL_ARGS[@]}"

stage "Done"
log "Refined manifest: $REFINED_MANIFEST"
log "Candidate rankings: $CANDIDATE_RANKINGS"
log "Metrics: $METRICS"
log "Annotation IoU: ${ANNOTATION_IOU_OUTPUT:-<disabled>}"
log "Score shard logs: $SCORE_SHARD_DIR"
