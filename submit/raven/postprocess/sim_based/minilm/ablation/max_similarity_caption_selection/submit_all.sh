#!/usr/bin/env bash
set -euo pipefail

taus=($(seq 0.15 0.02 0.95))

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
    tau_nodot="${tau/./}"
    log_stem="${log_dir}/sim_based_max_similarity_caption_selection_tau_${tau_nodot}"

    sbatch \
      --job-name "minilm_sim_based_max_similarity_caption_selection_tau_${tau}" \
      -o "${log_stem}_%A_%a_%x_%j_%N.out" \
      -e "${log_stem}_%A_%a_%x_%j_%N.err" \
      --export=ALL,TAU="${tau}" \
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/ablation/max_similarity_caption_selection/script.sbatch
done
