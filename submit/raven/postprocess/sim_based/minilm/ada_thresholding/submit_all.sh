#!/usr/bin/env bash
set -euo pipefail

# eta_bins=(10)
eta_bins=(8 10 12 16 20)
# kappa_counts=(7)
kappa_counts=(5 7 9 11 13)
# tau_consecutives=(5)
tau_consecutives=(3 5 7 9)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for eta_bin in "${eta_bins[@]}"; do
  for kappa_count in "${kappa_counts[@]}"; do
    for tau_consecutive in "${tau_consecutives[@]}"; do
      eta_bin_nodot="${eta_bin/./}"
      log_stem="${log_dir}/sim_based_eta_bin_${eta_bin_nodot}_kappa_count_${kappa_count}_tau_consecutive_${tau_consecutive}"
      sbatch \
        --job-name "minilm_sim_based_eta_bin_${eta_bin}_kappa_count_${kappa_count}_tau_consecutive_${tau_consecutive}" \
        -o "${log_stem}_%A_%a_%x_%j_%N.out" \
        -e "${log_stem}_%A_%a_%x_%j_%N.err" \
        --export=ALL,ETA_BIN="${eta_bin}",KAPPA_COUNT="${kappa_count}",TAU_CONSECUTIVE="${tau_consecutive}" \
        /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/ada_thresholding/script.sbatch
    done
  done
done
