#!/bin/bash

echo "Evaluation of original Ego4D dataset"
python3 /u/dduka/project/AVION/second_party/utils/evaluate_refinement.py --csv /u/dduka/project/AVION/second_party/utils/manual.csv --pkl /dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl

echo "Evaluation of scaled Ego4D dataset"
python3 /u/dduka/project/AVION/second_party/utils/evaluate_refinement.py --csv /u/dduka/project/AVION/second_party/utils/manual.csv --pkl /dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_with_uuid.pkl

echo "Evaluation of QWEN refined original Ego4D dataset"
python3 /u/dduka/project/AVION/second_party/utils/evaluate_refinement.py --csv /u/dduka/project/AVION/second_party/utils/manual.csv --pkl /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/standard/pickle/ego4d_train_standard_1_caption_vllm_with_uuid.pkl

echo "Evaluation of QWEN refined scaled Ego4D dataset"
python3 /u/dduka/project/AVION/second_party/utils/evaluate_refinement.py --csv /u/dduka/project/AVION/second_party/utils/manual.csv --pkl /dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/scaled/pickle/ego4d_train_scaled_1_caption_vllm_with_uuid.pkl

