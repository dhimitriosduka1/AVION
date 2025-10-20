import os
import json
import math
import random
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

from second_party.storage.sqlite import SQLiteClient
from second_party.preprocess.utils import preprocess_captions

random.seed(42)
np.random.seed(42)


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

            # Update the timestamp so that it is ofsetted by the chunk start
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


def resolve_embedding_at_idx(
    flattened_metadata: List[Dict[str, Any]],
    idx: int,
    embeddings_to_include: int,
    lavila_embeddings_client: SQLiteClient,
    seed: str = None,
) -> np.ndarray:
    captions = flattened_metadata[idx]["captions"]

    rng = random.Random(f"{seed}:{idx}" if seed is not None else str(idx))
    chosen = rng.sample(captions, embeddings_to_include)

    embeddings = []
    for caption in chosen:
        processed = preprocess_captions([caption])[0]
        embedding = lavila_embeddings_client.get_embedding(processed)
        embeddings.append(np.asarray(embedding, dtype=np.float32))

    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    return mean_embedding / np.linalg.norm(mean_embedding)


def expand_window(
    flattened_metadata: List[Dict[str, Any]],
    anchor_embedding: np.ndarray,
    anchor_idx: int,
    tau: float,
    lavila_embeddings_client: SQLiteClient,
    embeddings_to_include: int,
    seed: str,
) -> (float, float):
    """
    Expand from anchor_idx left and right while similarity >= tau.
    Uses get_idx_embedding(idx) to retrieve/calc embeddings with caching.
    """

    # Expand left
    left = anchor_idx
    while left - 1 >= 0:
        embedding = resolve_embedding_at_idx(
            flattened_metadata,
            left - 1,
            embeddings_to_include,
            lavila_embeddings_client,
            seed,
        )
        if cosine_sim(anchor_embedding, embedding) < tau:
            break
        left -= 1

    # Expand right
    right = anchor_idx
    while right + 1 < len(flattened_metadata):
        embedding = resolve_embedding_at_idx(
            flattened_metadata,
            right + 1,
            embeddings_to_include,
            lavila_embeddings_client,
            seed,
        )
        if cosine_sim(anchor_embedding, embedding) < tau:
            break
        right += 1

    return math.floor(flattened_metadata[left]["timestamps"][0]), math.ceil(
        flattened_metadata[right]["timestamps"][-1]
    )


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

    results = []
    for sample in tqdm(data, desc="Processing samples"):
        video_id, start, end, original_caption = sample
        anchor_timestamp = 0.5 * (start + end)

        caption = preprocess_captions([original_caption])[0]

        # The anchor caption against which the similarities are computed.
        anchor_caption = ego4d_embeddings.get_embedding(caption)

        flattened_metadata = load_all_chunks_metadata_for_video(
            args.chunk_metadata_root, video_id
        )

        anchor_idx = resolve_anchor_index(anchor_timestamp, flattened_metadata)

        new_start, new_end = expand_window(
            flattened_metadata,
            anchor_caption,
            anchor_idx,
            args.tau,
            lavila_embeddings,
            args.embeddings_to_include,
            video_id,
        )

        results.append((video_id, new_start, new_end, original_caption))

    with open(
        Path(args.output_path)
        / f"ego4d_train_tau_{args.tau}_embeddings_{args.embeddings_to_include}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} results to {args.output_path}")


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
    parser.add_argument(
        "--output-path", type=str, required=True, help="Pickle output file"
    )
    args = parser.parse_args()
    main(args)
