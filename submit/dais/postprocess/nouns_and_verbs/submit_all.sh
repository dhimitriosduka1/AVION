#!/usr/bin/env bash
set -euo pipefail

gaps=(0.5)

for gap in "${gaps[@]}"; do 
  gap_nodot="${gap/./}"
  sbatch \
    --job-name "nouns_and_verbs_gap_${gap_nodot}" \
    -o "nouns_and_verbs_gap_${gap_nodot}_%A_%x_%j_%N.out" \
    -e "nouns_and_verbs_gap_${gap_nodot}_%A_%x_%j_%N.err" \
    --export=ALL,GAP="${gap}" \
    /u/dduka/project/AVION/submit/dais/postprocess/nouns_and_verbs/script.sbatch
done