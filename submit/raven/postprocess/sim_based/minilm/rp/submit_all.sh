#!/usr/bin/env bash
set -euo pipefail

taus=(0.26)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
    tau_nodot="${tau/./}"
    log_stem="${log_dir}/sim_based_rp_tau_${tau_nodot}_nr_embeddings_10"

    sbatch \
      --job-name "minilm_sim_based_rp_tau_${tau}_nr_embeddings_10" \
      -o "${log_stem}_%A_%a_%x_%j_%N.out" \
      -e "${log_stem}_%A_%a_%x_%j_%N.err" \
      --export=ALL,TAU="${tau}" \
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/rp/script.sbatch
done
