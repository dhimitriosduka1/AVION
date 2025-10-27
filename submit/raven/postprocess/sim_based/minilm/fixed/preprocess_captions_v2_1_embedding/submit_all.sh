#!/usr/bin/env bash
set -euo pipefail

taus=(0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.40 0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.92 0.94 0.95)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
  tau_nodot="${tau/./}"
  log_stem="${log_dir}/sim_based_tau_${tau_nodot}_nr_embeddings_10"

  sbatch \
    --job-name "minilm_sim_based_tau_${tau}_nr_embeddings_10" \
    -o "${log_stem}_%A_%a_%x_%j_%N.out" \
    -e "${log_stem}_%A_%a_%x_%j_%N.err" \
    --export=ALL,TAU="${tau}" \
    /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/fixed/preprocess_captions_v2_1_embedding/script.sbatch
done
