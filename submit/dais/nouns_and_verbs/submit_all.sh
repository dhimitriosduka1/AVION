#!/usr/bin/env bash
set -euo pipefail

gaps=(1.1 1.4 1.7 2.0 2.3 2.6 2.9 3.0 4.0)

for gap in "${gaps[@]}"; do 
  gap_nodot="${gap/./}"
  sbatch \
    --job-name "4_gpu_nouns_and_verbs_gap_${gap_nodot}" \
    -o "4_gpu_nouns_and_verbs_gap_${gap_nodot}_%A_%a_%x_%j_%N.out" \
    -e "4_gpu_nouns_and_verbs_gap_${gap_nodot}_%A_%a_%x_%j_%N.err" \
    --export=ALL,GAP="${gap}" \
    /u/dduka/project/AVION/submit/dais/nouns_and_verbs/script.sbatch
done