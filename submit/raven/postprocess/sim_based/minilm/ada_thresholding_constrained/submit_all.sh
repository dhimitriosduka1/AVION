#!/usr/bin/env bash
set -euo pipefail

keep_thresholds=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for keep_threshold in "${keep_thresholds[@]}"; do
  keep_threshold_nodot="${keep_threshold/./}"
  log_stem="${log_dir}/sim_based_keep_threshold_${keep_threshold_nodot}"
  sbatch \
    --job-name "minilm_sim_based_keep_threshold_${keep_threshold}" \
    -o "${log_stem}_%A_%a_%x_%j_%N.out" \
    -e "${log_stem}_%A_%a_%x_%j_%N.err" \
    --export=ALL,KEEP_THRESHOLD="${keep_threshold}" \
    /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/ada_thresholding_constrained/script.sbatch
done
