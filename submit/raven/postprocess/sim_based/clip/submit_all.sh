#!/usr/bin/env bash
set -euo pipefail

taus=(0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
  tau_nodot="${tau/./}"
  log_stem="${log_dir}/sim_based_tau_${tau_nodot}_nr_embeddings_10"

  sbatch \
    --job-name "clip_sim_based_tau_${tau}_nr_embeddings_10" \
    -o "${log_stem}_%A_%a_%x_%j_%N.out" \
    -e "${log_stem}_%A_%a_%x_%j_%N.err" \
    --export=ALL,TAU="${tau}" \
    /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/clip/script.sbatch
done
