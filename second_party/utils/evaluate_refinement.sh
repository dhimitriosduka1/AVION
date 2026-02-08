#!/bin/bash

echo "Evaluation of original Ego4D dataset"
python3 /u/dduka/project/AVION/second_party/utils/evaluate_refinement.py \
    --csv /u/dduka/project/AVION/second_party/utils/manual.csv \
    --pkl /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl \
    /dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_with_uuid.pkl \
    /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_standard_1_caption_vllm_with_uuid.pkl \
    /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/scaled/pickle/ego4d_train_scaled_1_caption_vllm_with_uuid.pkl \
    /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_standard_10_caption_vllm_with_uuid.pkl \
    /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_ttrv_strandard_standard_400_step_with_uuid.pkl \
    /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl \
    --names "Original Ego4D" "2.1x Scaled Ego4D" "vLLM Original Ego4D 1 Caption" "vLLM Scaled Ego4D 1 Caption" "vLLM Original Ego4D 10 Captions" "TTRV Standard Standard 400 Step" "Deduplicated Ego4D" \
    --include_scaled \
    --video_lengths /dais/fs/scratch/dduka/databases/ego4d/video_lengths.json \
    --min_scale 1.0 \
    --max_scale 3.0 \