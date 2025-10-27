#!/usr/bin/env bash
set -euo pipefail

taus=(0.27 0.29 0.31 0.33 0.35)
numbers_of_segments=(1 2 3 4 5 6 7 8 9 10)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
  for num_segments in "${numbers_of_segments[@]}"; do
    tau_nodot="${tau/./}"
    log_stem="${log_dir}/sim_based_nr_of_segments_tau_${tau_nodot}_num_segments_${num_segments}"

    sbatch \
      --job-name "minilm_sim_based_nr_of_segments_tau_${tau}_num_segments_${num_segments}" \
      -o "${log_stem}_%A_%a_%x_%j_%N.out" \
      -e "${log_stem}_%A_%a_%x_%j_%N.err" \
      --export=ALL,TAU="${tau}",NUM_SEGMENTS="${num_segments}" \
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/ablation/nr_of_segments/script.sbatch
  done
done
