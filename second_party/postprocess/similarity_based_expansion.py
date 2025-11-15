import argparse
import pickle
import os
import json
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm
from second_party.preprocess.utils import (
    preprocess_caption_v2 as preprocess_captions,
)

def group_samples_by_video(data):
    video_groups = defaultdict(list)
    for idx, sample in enumerate(data):
        video_id, start, end, original_caption = sample
        video_groups[video_id].append((idx, start, end, original_caption))
    return video_groups

def process_one_video(video_id, gt_segments, ego4d_embeddings, ego4d_captions, lavila_embeddings, lavila_captions, tau):
    
    pass
    

def main(args):
    wandb.init(
        project="Thesis",
        name=f"Similarity Based Expansion",
        config={**args.__dict__},
        group=f"Similarity Based Expansion",
    )

    # Read the dataset
    with open(args.dataset, "rb") as f:
        gt_dataset = pickle.load(f)

    # Group samples by video_id
    gt_dataset = group_samples_by_video(gt_dataset)

    ego4d_embeddings_shape = json.load(
        open(os.path.join(args.dataset_embeddings_path, "shape.json"), "r")
    )
    ego4d_unique_captions = json.load(
        open(os.path.join(args.dataset_embeddings_path, "captions.json"), "r")
    )
    ego4d_embeddings = np.memmap(
        os.path.join(args.dataset_embeddings_path, "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(ego4d_embeddings_shape["shape"]),
    )

    lavila_embeddings_shape = json.load(
        open(os.path.join(args.lavila_embeddings_path, "shape.json"), "r")
    )
    lavila_unique_captions = json.load(
        open(os.path.join(args.lavila_embeddings_path, "captions.json"), "r")
    )
    lavila_embedding = np.memmap(
        os.path.join(args.lavila_embeddings_path, "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(lavila_embeddings_shape["shape"]),
    )

    results = []
    for video_id, samples in tqdm(gt_dataset.items(), desc="Processing videos"):

        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess AVION retrieval results with similarity-based expansion"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Pickle file with samples: (video_id, start, end, caption)",
    )
    parser.add_argument(
        "--dataset-embeddings-path",
        type=str,
        required=True,
        help="Precomputed and normalized Ego4D captions embeddings",
    )
    parser.add_argument(
        "--lavila-embeddings-path",
        type=str,
        required=True,
        help="Precomputed and normalized LaViLa captions embeddings",
    )
    parser.add_argument(
        "--chunk-metadata-root",
        type=str,
        required=True,
        help="Root folder containing per-chunk captions metadata JSONs",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Embedding model used to extract embeddings",
    )
    parser.add_argument(
        "--threashold", type=float, default=0.7, help="Cosine similarity threshold"
    )

    args = parser.parse_args()
    main(args)