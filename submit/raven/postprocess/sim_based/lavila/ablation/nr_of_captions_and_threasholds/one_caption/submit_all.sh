#!/usr/bin/env bash
set -euo pipefail

# Taus should start from 0.15 and end at 0.95. Define it in a loop or smth, not hardcoded.
taus=($(seq 0.15 0.02 0.95))
numbers_of_segments=(1)

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
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/lavila/ablation/nr_of_captions_and_threasholds/one_caption/script.sbatch
  done
done
