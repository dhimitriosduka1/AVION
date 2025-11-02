#!/usr/bin/env bash
set -euo pipefail

taus=(0.27 0.29 0.31 0.33 0.35)
numbers_of_captions=(1 2 3 4 5 6 7 8 9 10)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
  for num_captions in "${numbers_of_captions[@]}"; do
    tau_nodot="${tau/./}"
    log_stem="${log_dir}/sim_based_nr_of_captions_tau_${tau_nodot}_num_captions_${num_captions}"

    sbatch \
      --job-name "minilm_sim_based_nr_of_captions_tau_${tau}_num_captions_${num_captions}" \
      -o "${log_stem}_%A_%a_%x_%j_%N.out" \
      -e "${log_stem}_%A_%a_%x_%j_%N.err" \
      --export=ALL,TAU="${tau}",NUM_CAPTIONS="${num_captions}" \
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/ablation/anchor_mode_captions_mean_pooling/script.sbatch
  done
done
