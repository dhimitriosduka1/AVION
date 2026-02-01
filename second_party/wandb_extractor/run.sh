#!/bin/bash

cd /u/dduka/project/AVION/second_party/wandb_extractor/

python3 download_wandb_metrics.py \
  --entity dduka-max-planck-society \
  --project Thesis \
  --run-ids DAIS_DUAL_ENC_BASELINE \
  --main-metric test_ego4d_mir_avg_map \
  --output-dir /u/dduka/project/AVION/second_party/wandb_extractor/output 