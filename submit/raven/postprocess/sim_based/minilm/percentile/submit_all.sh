#!/usr/bin/env bash
set -euo pipefail

taus=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
  tau_nodot="${tau/./}"
  log_stem="${log_dir}/sim_based_percentile_tau_${tau_nodot}_nr_embeddings_10"

  sbatch \
    --job-name "minilm_sim_based_percentile_tau_${tau}_nr_embeddings_10" \
    -o "${log_stem}_%A_%a_%x_%j_%N.out" \
    -e "${log_stem}_%A_%a_%x_%j_%N.err" \
    --export=ALL,TAU="${tau}" \
    /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/percentile/script.sbatch
done
