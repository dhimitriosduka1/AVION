import os
import json
import math
import wandb
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from second_party.storage.sqlite import SQLiteClient
from second_party.preprocess.utils import preprocess_captions


def cosine_sim(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    Cosine similarity assuming embeddings are already normalized upstream.
    """
    return float(np.dot(embeddings1, embeddings2))


def load_all_chunks_metadata_for_video(chunk_metadata_root: str, video_id: str):
    base_video_path = Path(chunk_metadata_root) / video_id
    captions = []
    for chunk_path in base_video_path.glob("*.mp4/captions.json"):
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_start_offset = int(str(chunk_path).split("/")[-2].split(".")[0])
            metadata = json.load(f)["metadata"]
            # Update the timestamp so that it is offsetted by the chunk start
            for m in metadata:
                m["timestamps"] = [x + chunk_start_offset for x in m["timestamps"]]
            captions.extend(metadata)
    # Sort by start timestamp
    captions.sort(key=lambda x: x["timestamps"][0])
    return captions


def resolve_anchor_index(anchor_timestamp: float, flattened_metadata: Any) -> int:
    for idx, metadata in enumerate(flattened_metadata):
        start = math.floor(metadata["timestamps"][0])
        end = math.ceil(metadata["timestamps"][-1])
        if start <= anchor_timestamp < end:
            return idx
    raise ValueError("Anchor timestamp not found in flattened metadata")


def precompute_video_embeddings(
    flattened_metadata: List[Dict[str, Any]],
    embeddings_to_include: int,
    lavila_embeddings_client: SQLiteClient,
    seed: str,
) -> np.ndarray:
    """
    Precompute all embeddings for a video at once.
    Returns: array of shape (num_segments, embedding_dim)
    """
    all_embeddings = []

    for idx in range(len(flattened_metadata)):
        captions = flattened_metadata[idx]["captions"]
        rng = random.Random(f"{seed}:{idx}")
        chosen = rng.sample(captions, min(embeddings_to_include, len(captions)))

        embeddings = []
        for caption in chosen:
            processed = preprocess_captions([caption])[0]
            embedding = lavila_embeddings_client.get_embedding(processed)
            embeddings.append(np.asarray(embedding, dtype=np.float32))

        mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
        normalized = mean_embedding / np.linalg.norm(mean_embedding)
        all_embeddings.append(normalized)

    return np.stack(all_embeddings, axis=0)


def expand_window(
    precomputed_embeddings: np.ndarray,
    flattened_metadata: List[Dict[str, Any]],
    anchor_embedding: np.ndarray,
    anchor_idx: int,
    tau: float,
) -> Tuple[float, float]:
    """
    Expand from anchor_idx left and right while similarity >= tau.
    Uses precomputed embeddings array.
    """
    # Expand left
    left = anchor_idx
    while left - 1 >= 0:
        embedding = precomputed_embeddings[left - 1]
        if cosine_sim(anchor_embedding, embedding) < tau:
            break
        left -= 1

    # Expand right
    right = anchor_idx
    while right + 1 < len(flattened_metadata):
        embedding = precomputed_embeddings[right + 1]
        if cosine_sim(anchor_embedding, embedding) < tau:
            break
        right += 1

    return math.floor(flattened_metadata[left]["timestamps"][0]), math.ceil(
        flattened_metadata[right]["timestamps"][-1]
    )


def group_samples_by_video(
    data: List[Tuple],
) -> Tuple[Dict[str, List[Tuple]], List[int]]:
    """
    Group samples by video_id for batch processing.
    Returns: (video_groups, original_indices) where original_indices maps back to input order.
    """
    video_groups = defaultdict(list)
    for idx, sample in enumerate(data):
        video_id, start, end, original_caption = sample
        video_groups[video_id].append((idx, start, end, original_caption))
    return video_groups


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"
    assert args.ego4d_embeddings_path.endswith(
        ".sqlite"
    ), "Ego4d embeddings must be a sqlite file"
    assert args.lavila_embeddings_path.endswith(
        ".sqlite"
    ), "LaViLa embeddings must be a sqlite file"

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project="Thesis",
        name=f"Threshold {args.tau} - Embeddings Number {args.embeddings_to_include} - Temperature {args.temperature}",
        config={**args.__dict__},
        group=f"Similarity Based Timestamp Shifting - {args.embedding_model}",
    )

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

    # Group samples by video
    video_groups = group_samples_by_video(data)
    print(f"Processing {len(video_groups)} unique videos")

    # Results will be stored with their original index
    results_dict = {}

    # Process one video at a time
    for i, (video_id, samples) in enumerate(
        tqdm(enumerate(video_groups.items()), desc="Processing videos")
    ):
        # Load metadata once per video
        flattened_metadata = load_all_chunks_metadata_for_video(
            args.chunk_metadata_root, f"{video_id}.mp4"
        )

        # Precompute all embeddings for this video once
        precomputed_embeddings = precompute_video_embeddings(
            flattened_metadata,
            args.embeddings_to_include,
            lavila_embeddings,
            video_id,
        )

        # Process all samples for this video
        for original_idx, start, end, original_caption in samples:
            anchor_timestamp = 0.5 * (start + end)
            caption = preprocess_captions([original_caption])[0]

            # The anchor caption against which the similarities are computed
            anchor_caption = ego4d_embeddings.get_embedding(caption)

            anchor_idx = resolve_anchor_index(anchor_timestamp, flattened_metadata)

            new_start, new_end = expand_window(
                precomputed_embeddings,
                flattened_metadata,
                anchor_caption,
                anchor_idx,
                args.tau,
            )

            results_dict[original_idx] = (
                video_id,
                new_start,
                new_end,
                original_caption,
            )

        # Throttle W&B logging
        if (i % args.log_every) == 0:
            wandb.log({"progress": (i + 1) / len(video_groups)}, step=i + 1)

    # Reconstruct results in original order
    results = [results_dict[i] for i in range(len(data))]

    os.makedirs(Path(args.output_path) / args.embedding_model, exist_ok=True)

    output_file = (
        Path(args.output_path)
        / args.embedding_model
        / f"ego4d_train_temperature_{args.temperature}_threshold_{args.tau}_embeddings_{args.embeddings_to_include}.pkl"
    )
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} results to {output_file}")

    wandb.finish()


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
        "--embedding-model",
        type=str,
        required=True,
        help="Embedding model used to extract embeddings",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature used to generate captions with LaViLa model",
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
    parser.add_argument(
        "--output-path", type=str, required=True, help="Pickle output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-every", type=int, default=100, help="Log every n samples"
    )
    args = parser.parse_args()
    main(args)
