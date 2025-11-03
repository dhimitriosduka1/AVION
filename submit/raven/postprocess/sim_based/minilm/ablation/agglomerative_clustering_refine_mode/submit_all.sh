#!/usr/bin/env bash
set -euo pipefail

taus=($(seq 0.15 0.02 0.70))

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
    tau_nodot="${tau/./}"
    log_stem="${log_dir}/sim_based_tau_${tau_nodot}_refine_mode_agglomerative_clustering"

    sbatch \
      --job-name "minilm_sim_based_tau_${tau}_refine_mode_agglomerative_clustering" \
      -o "${log_stem}_%A_%a_%x_%j_%N.out" \
      -e "${log_stem}_%A_%a_%x_%j_%N.err" \
      --export=ALL,TAU="${tau}" \
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/ablation/agglomerative_clustering_refine_mode/script.sbatch
done
