import os
import json
import math
import torch
import random
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from typing import List, Dict, Any, Optional

from second_party.storage.sqlite import SQLiteClient
from second_party.preprocess.utils import preprocess_captions

random.seed(42)
np.random.seed(42)


def resolve_video_chunk_path(
    video_id: str, start: float, end: float, chunk_size: int = 15
) -> List[str]:
    """
    Return the list of metadata file paths for chunks overlapping the interval [start, end).

    Chunks are chunk_size seconds each: [0, chunk_size), [chunk_size, 2*chunk_size), ...

    Examples (chunk_size=15):
      - start=2.3, end=5.4   -> chunk starting at 0
      - start=10.3, end=16.9 -> chunks starting at 0 and 15 (crosses the 15s boundary)

    NOTE: This function returns metadata paths following the layout:
          f"{video_id}.mp4/{chunk_start}.mp4/captions.json"
    """
    if start < 0 or end < 0:
        raise ValueError("Start and end must be non-negative values")
    if end <= start:
        raise ValueError("End must be greater than start value")

    epsilon = 1e-9
    end_adj = max(0.0, end - epsilon)

    start_idx = int(math.floor(start / chunk_size))
    end_idx = int(math.floor(end_adj / chunk_size))

    paths = []
    for idx in range(start_idx, end_idx + 1):
        chunk_start_time = int(idx * chunk_size)
        chunk_filename = f"{chunk_start_time}"
        paths.append(f"{video_id}.mp4/{chunk_filename}.mp4/captions.json")

    return paths


def get_chunks_metadata(chunk_metadata_root, paths):
    return [json.load(open(os.path.join(chunk_metadata_root, path))) for path in paths]


def resolve_metadata_idx_based_on_anchor_timestamp(anchor_timestamp: float) -> int:
    """
    For current implementation (1 segment per second), the index is simply floor(anchor_timestamp).
    """  # for i, m in enumerate(metadata):
    #     start, end = m["timestamps"][0], m["timestamps"][-1]
    #     if start <= anchor_timestamp < end:
    #         return i
    # return None
    # For the current implementation, where we have 1 segment per each second, finding
    # the correct index is as easy as:
    return math.floor(anchor_timestamp)


def resolve_embedding_at_idx(
    metadata: List[Dict[str, Any]],
    idx: int,
    embeddings_to_include: int,
    lavila_embeddings_client: SQLiteClient,
    seed: str = None,
) -> np.ndarray:
    """Average N sampled caption embeddings for metadata[idx] (deterministic by seed+idx)."""
    caps = metadata[idx]["captions"]
    if embeddings_to_include > len(caps):
        raise ValueError("embeddings_to_include exceeds available captions")

    rng = random.Random(f"{seed}:{idx}" if seed is not None else str(idx))
    chosen = rng.sample(caps, embeddings_to_include)

    embeddings = []
    for caption in chosen:
        processed = preprocess_captions([caption])[0]
        embedding = lavila_embeddings_client.get_embedding(processed)
        embeddings.append(np.asarray(embedding, dtype=np.float32))

    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    return mean_embedding / np.linalg.norm(mean_embedding)


def cosine_sim(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    Cosine similarity assuming embeddings are already normalized upstream.
    """
    return float(np.dot(embeddings1, embeddings2))


def expand_window(
    metadata: List[Dict[str, Any]],
    anchor_caption_embedding: np.ndarray,
    anchor_idx: int,
    tau: float,
    lavila_embeddings_client: SQLiteClient,
    embeddings_to_include: int,
) -> (float, float):
    """
    Expand from anchor_idx left and right while similarity >= tau.
    Uses get_idx_embedding(idx) to retrieve/calc embeddings with caching.
    """
    n = len(metadata)
    if n == 0:
        raise ValueError("Empty metadata timeline; cannot expand window.")

    # Expand left
    left = anchor_idx
    while left - 1 >= 0:
        emb_left = resolve_embedding_at_idx(
            metadata, left - 1, embeddings_to_include, lavila_embeddings_client
        )
        if cosine_sim(anchor_caption_embedding, emb_left) < tau:
            break
        left -= 1

    # Expand right
    right = anchor_idx
    while right + 1 < n:
        emb_right = resolve_embedding_at_idx(
            metadata, right + 1, embeddings_to_include, lavila_embeddings_client
        )
        if cosine_sim(anchor_caption_embedding, emb_right) < tau:
            break
        right += 1

    new_start = metadata[left]["timestamps"][0]
    new_end = metadata[right]["timestamps"][-1]
    return new_start, new_end


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"
    assert args.ego4d_embeddings_path.endswith(
        ".sqlite"
    ), "Ego4d embeddings must be a sqlite file"
    assert args.lavila_embeddings_path.endswith(
        ".sqlite"
    ), "LaViLa embeddings must be a sqlite file"

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    print(f"Opening {args.ego4d_embeddings_path} embeddings")
    ego4d_embeddings = SQLiteClient(args.ego4d_embeddings_path)
    print(f"Loaded {ego4d_embeddings.count_embeddings()} ego4d embeddings")

    print(f"Opening {args.lavila_embeddings_path} embeddings")
    lavila_embeddings = SQLiteClient(args.lavila_embeddings_path)
    print(f"Loaded {lavila_embeddings.count_embeddings()} lavila embeddings")

    for sample in tqdm(data, desc="Processing samples"):
        video_id, start, end, caption = sample
        anchor_timestamp = 0.5 * (start + end)

        caption = preprocess_captions([caption])[0]

        # The anchor caption against which the similarities are computed.
        anchor_caption = ego4d_embeddings.get_embedding(caption)

        video_chunks_paths = resolve_video_chunk_path(
            video_id, start, end, args.chunk_size
        )

        video_chunks_metadata = get_chunks_metadata(
            args.chunk_metadata_root, video_chunks_paths
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Pickle file with samples: (video_id, start, end, caption)",
    )
    parser.add_argument(
        "--ego4d-embeddings-path",
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
        "--tau", type=float, default=0.7, help="Cosine similarity threshold"
    )
    parser.add_argument(
        "--embeddings-to-include",
        type=int,
        default=4,
        help="Number of embeddings to include for each segment",
    )
    parser.add_argument("--chunk-size", type=int, default=15, help="Seconds per chunk")
    parser.add_argument(
        "--video-root",
        type=str,
        default="",
        help="Root directory for videos",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Pickle output file"
    )
    args = parser.parse_args()
    main(args)
