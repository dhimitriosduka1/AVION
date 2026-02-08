#!/bin/bash

python3 /u/dduka/project/AVION/second_party/qwen3vl/merge_results.py \
    --video-len-path /dais/fs/scratch/dduka/databases/ego4d/video_lengths.json \
    --json-path /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/temp/ \
    --output-file /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_ttrv_strandard_standard_400_step_with_uuid.pkl \
    --num-of-captions 1 \
    --original-ego4d-path /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl \