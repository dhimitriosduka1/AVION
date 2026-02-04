#!/bin/bash

cd /u/dduka/work/projects/Thesis/AVION

uv run /u/dduka/work/projects/Thesis/AVION/second_party/wandb_extractor/download_wandb_metrics.py \
  --entity dduka-max-planck-society \
  --project Thesis \
  --run-ids DAIS_DUAL_ENC_BASELINE_362K DAIS_DUAL_ENC_2.1x_362K DAIS_DUAL_ENC_1_CAP_STANDARD_362K DAIS_DUAL_ENC_10_CAP_STANDARD_362K DAIS_DUAL_ENC_1_CAP_SCALED_362K \
  --main-metric test_ego4d_mir_avg_map \
  --output-dir /u/dduka/work/projects/Thesis/AVION/second_party/wandb_extractor/output