import math
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import wandb

import numpy as np
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine GT segments by optimally assigning in-between pseudo-labels without crossing GTs."
    )
    parser.add_argument(
        "--dataset",
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_enriched.pkl",
        help="Path to ground-truth pickle (e.g., ego4d_train.pkl)",
    )
    parser.add_argument(
        "--pseudolabels",
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_uncovered_all.narrator_63690737.return_5_enriched.pkl",
        help="Path to pseudo-label pickle",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Desired output path",
    )
    parser.add_argument(
        "--gap",
        default=1.5,
        type=float,
        help="Max allowable gap (seconds) when chaining adjacent segments",
    )
    parser.add_argument(
        "--use",
        default="both",
        choices=["both", "nouns", "verbs"],
        help="Whether to use only nouns or verbs for the overlap check",
    )
    return parser.parse_args()


def load_data(path):
    print(f"Loading data from {path} ...")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def save_data(path, data):
    print(f"\nSaving refined data to {path} ...")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _group_videos_by_video_id(data):
    video_groups = defaultdict(list)
    for sample in data:
        video_id = sample[0]
        video_groups[video_id].append(sample)
    for video_id in video_groups:
        video_groups[video_id].sort(key=lambda x: float(x[1]))
    return video_groups


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _merge(source, target):
    merged_data = {}
    for video_id, segments in source.items():
        merged_data[video_id] = segments + target[video_id]

    for video_id in merged_data:
        merged_data[video_id].sort(key=lambda x: float(x[1]))
    
    return merged_data

def main(args):
    wandb.init(
        project="Thesis",
        name=f"Refine Dataset - Gap {args.gap}s",
        config={**args.__dict__},
        group=f"Refine Dataset",
    )

    ground_truth_data = load_data(args.dataset)
    pseudo_labels_data = load_data(args.pseudolabels)

    total = len(ground_truth_data)
    total_pl = len(pseudo_labels_data)
    print(f"Total ground truth segments: {total}")
    print(f"Total pseudo-labels segments: {total_pl}")

    print("Grouping videos by video_id...")
    ground_truth_video_groups = _group_videos_by_video_id(ground_truth_data)
    pseudo_labels_video_groups = _group_videos_by_video_id(pseudo_labels_data)

    print("Merging original and pseudo-labels...")
    data = _merge(ground_truth_video_groups, pseudo_labels_video_groups)

    

if __name__ == "__main__":
    args = parse_args()
    main(args)
