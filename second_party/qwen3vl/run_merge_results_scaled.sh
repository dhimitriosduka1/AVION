#!/bin/bash

python3 /u/dduka/work/projects/Thesis/AVION/second_party/qwen3vl/merge_results.py \
    --video-len-path /ptmp/dduka/databases/ego4d/video_lengths.json \
    --json-path /ptmp/dduka/databases/ego4d/qwen_refinement/standard/ \
    --output-file /ptmp/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_standard_1_caption_padding_1_vllm_with_uuid.pkl \
    --num-of-captions 1 \
    --original-ego4d-path /ptmp/dduka/databases/ego4d/ego4d_train_with_uuid.pkl \