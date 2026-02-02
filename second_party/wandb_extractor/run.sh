#!/bin/bash

cd /u/dduka/project/AVION/second_party/wandb_extractor/

python3 download_wandb_metrics.py \
  --entity dduka-max-planck-society \
  --project Thesis \
  --run-ids DAIS_DUAL_ENC_BASELINE DAIS_DUAL_ENC_MULT_2.1 DAIS_DUAL_ENC_1_CAPTION_QWEN3VL DAIS_DUAL_ENC_1_CAPTION_SCALED_QWEN3VL DAIS_DUAL_ENC_10_CAPTION_QWEN3VL DAIS_LAVILA_BASELINE DAIS_LAVILA_x2.1_SCALED DAIS_LAVILA_REFINED_QWEN_1_CAPTION_STANDARD DAIS_LAVILA_REFINED_QWEN_1_CAPTION_SCALED DAIS_LAVILA_REFINED_QWEN_10_CAPTION_STANDARD \
  --main-metric test_ego4d_mir_avg_map \
  --output-dir /u/dduka/project/AVION/second_party/wandb_extractor/output 