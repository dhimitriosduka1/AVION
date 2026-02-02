#!/bin/bash -l

python3 /u/dduka/project/AVION/second_party/utils/copy_timestamps_to_lavila_dataset.py \
    --source_timestamps "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset_2.1x_with_uuid.pkl" \
    --lavila_rephrased "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_with_uuid.pkl" \
    --out_with_uuid "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_refined_362k_subset_2.1x_scaled_with_uuid.pkl" \
    --out_refined "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_refined_362k_subset_2.1x_scaled.pkll"