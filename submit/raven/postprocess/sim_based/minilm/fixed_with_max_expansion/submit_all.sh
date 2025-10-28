#!/usr/bin/env bash
set -euo pipefail

taus=(0.26)
max_expansion_ratios=(1.5)

# Base log directory
log_dir="/ptmp/dduka/work/logs/avion"

for tau in "${taus[@]}"; do
    for max_expansion_ratio in "${max_expansion_ratios[@]}"; do
    tau_nodot="${tau/./}"
        log_stem="${log_dir}/sim_based_fixed_tau_${tau_nodot}_max_expansion_ratio_${max_expansion_ratio}"

        sbatch \
            --job-name "minilm_sim_based_fixed_tau_${tau}_max_expansion_ratio_${max_expansion_ratio}" \
            -o "${log_stem}_%A_%a_%x_%j_%N.out" \
            -e "${log_stem}_%A_%a_%x_%j_%N.err" \
            --export=ALL,TAU="${tau}",MAX_EXPANSION_RATIO="${max_expansion_ratio}" \
            /u/dduka/work/projects/Thesis/AVION/submit/raven/postprocess/sim_based/minilm/fixed_with_max_expansion/script.sbatch
    done
done
