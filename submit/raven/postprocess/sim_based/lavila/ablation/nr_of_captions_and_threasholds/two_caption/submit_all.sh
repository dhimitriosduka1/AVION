#!/usr/bin/env bash
set -euo pipefail

# Taus should start from 0.15 and end at 0.95. Define it in a loop or smth, not hardcoded.
taus=($(seq 0.15 0.02 0.95))

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
    tau_nodot="${tau/./}"
    log_stem="${log_dir}/sim_based_tau_${tau_nodot}_nr_of_captions_2"

    sbatch \
      --job-name "lavila_sim_based_tau_${tau}_nr_of_captions_2" \
      -o "${log_stem}_%A_%a_%x_%j_%N.out" \
      -e "${log_stem}_%A_%a_%x_%j_%N.err" \
      --export=ALL,TAU="${tau}" \
      /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/lavila/ablation/nr_of_captions_and_threasholds/two_caption/script.sbatch
done
