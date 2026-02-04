#!/bin/bash

python3 /u/dduka/work/projects/Thesis/AVION/second_party/qwen3vl/merge_results_multiple_responses.py \
    --video-len-path /ptmp/dduka/databases/ego4d/video_lengths.json \
    --json-path /ptmp/dduka/databases/ego4d/qwen_refinement/scaled/ \
    --output-file /ptmp/dduka/databases/ego4d/qwen_refinement/scaled/pickle/ego4d_train_scaled_10_caption_vllm_with_uuid.pkl \
    --num-of-captions 10 \
    --original-ego4d-path /ptmp/dduka/databases/ego4d/ego4d_train_random_shift_2.1_2.1_1.0_with_uuid.pkl \