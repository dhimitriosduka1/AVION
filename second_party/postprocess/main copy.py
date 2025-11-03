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
from typing import List, Dict, Any, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from second_party.postprocess.utils import (
    load_all_chunks_metadata_for_video, 
    resolve_anchor_index, 
    compute_intersection_stats, 
    cosine_sim, 
    plot_distribution, 
    group_samples_by_video,
    _WORKER,
    _init_worker,
)

def precompute_video_embeddings(
    flattened_metadata: List[Dict[str, Any]],
    lavila_embeddings: np.ndarray,
    lavila_unique_captions: Dict[str, int],
    preprocess_captions: Callable,
) -> np.ndarray:
    """
    Precompute all embeddings for a video at once.
    Returns: array of shape (num_segments, embedding_dim)
    """
    all_embeddings = []

    for idx in range(len(flattened_metadata)):
        captions = flattened_metadata[idx]["captions"]

        embeddings = []
        for caption in captions:
            processed = preprocess_captions([caption])[0]
            resolved_index = lavila_unique_captions[processed]
            embedding = lavila_embeddings[resolved_index]
            embeddings.append(embedding)
        
        # Compute the mean embedding of the captions
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        # Choose the embeddings to include: the most similar to the mean embedding
        chosen_embedding_index = np.argmax([cosine_sim(mean_embedding, embedding) for embedding in embeddings])
        chosen_embedding = lavila_embeddings[chosen_embedding_index]

        all_embeddings.append(chosen_embedding)

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

    segment_embeddings = precomputed_embeddings[left : right + 1]
    similarity_matrix = np.dot(segment_embeddings, segment_embeddings.T)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    mean_similarity_between_segments = np.mean(similarity_matrix[triu_indices])
    
    return (
        math.floor(flattened_metadata[left]["timestamps"][0]),
        math.ceil(flattened_metadata[right]["timestamps"][-1]),
        mean_similarity_between_segments,
    )

def _process_one_video(payload: Tuple[str, List[Tuple]], preprocess_captions: Callable):
    """
    Worker: process one video's samples.
    payload: (video_id, samples) where samples = [(original_idx, start, end, original_caption), ...]
    Returns: (results_dict_for_video, stats_dict_for_video)
    """
    video_id, samples = payload
    args = _WORKER["args"]

    # Load metadata and precompute embeddings for this video
    flattened_metadata = load_all_chunks_metadata_for_video(
        args["chunk_metadata_root"], f"{video_id}.mp4"
    )

    precomputed_embeddings = precompute_video_embeddings(
        flattened_metadata=flattened_metadata,
        lavila_embeddings=_WORKER["lavila_embeddings"],
        lavila_unique_captions=_WORKER["lavila_unique_captions"],
        preprocess_captions=preprocess_captions,
    )

    ego4d_embeddings = _WORKER["ego4d_embeddings"]
    ego4d_unique_captions = _WORKER["ego4d_unique_captions"]

    local_results: Dict[int, Tuple[str, float, float, str]] = {}
    local_stats = {
        "old_timestamps_duration": [],
        "new_timestamps_duration": [],
        "iou": [],
        "expansion_ratio": [],
        "mean_similarity_between_segments": [],
    }

    for original_idx, start, end, original_caption in samples:
        anchor_timestamp = 0.5 * (start + end)
        caption = preprocess_captions([original_caption])[0]

        # Resolve anchor caption -> ego4d embedding
        resolved_index = ego4d_unique_captions[caption]
        anchor_caption = ego4d_embeddings[resolved_index]

        try:
            anchor_idx = resolve_anchor_index(
                video_id, anchor_timestamp, flattened_metadata
            )

            new_start, new_end, mean_similarity_between_segments = expand_window(
                precomputed_embeddings,
                flattened_metadata,
                anchor_caption,
                anchor_idx,
                args["tau"],
            )
        except ValueError as e:
            print(f"Error resolving anchor index for video {video_id}: {e}")

            print(f"Flattened metadata: {flattened_metadata}")
            print(f"start: {start}, end: {end}")
            print(f"anchor_timestamp: {anchor_timestamp}")
            new_start = start
            new_end = end
            mean_similarity_between_segments = None

        local_results[original_idx] = (video_id, new_start, new_end, original_caption)

        local_stats["old_timestamps_duration"].append(end - start)
        local_stats["new_timestamps_duration"].append(new_end - new_start)

        s = compute_intersection_stats(start, end, new_start, new_end)
        local_stats["iou"].append(s["iou"])
        local_stats["expansion_ratio"].append(s["expansion_ratio"])

        if mean_similarity_between_segments is not None:
            local_stats["mean_similarity_between_segments"].append(
                mean_similarity_between_segments
            )

    return local_results, local_stats


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"

    if args.preprocess_function == "preprocess_captions":
        from second_party.preprocess.utils import preprocess_captions
    elif args.preprocess_function == "preprocess_caption_v2":
        from second_party.preprocess.utils import (
            preprocess_caption_v2 as preprocess_captions,
        )
    else:
        raise ValueError(f"Invalid preprocess function: {args.preprocess_function}")

    # Reproducibility for any rng used in parent
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project="Thesis",
        name=f"Threshold {args.tau} - Temperature {args.temperature} - Fn {args.preprocess_function} - ",
        config={**args.__dict__},
        group=f"Similarity Based Timestamp Shifting - {args.embedding_model}",
    )

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    # Optional: open shapes in parent to report counts (workers open real memmaps)
    print(f"Opening {args.ego4d_embeddings_path} embeddings")
    ego4d_embeddings_shape = json.load(
        open(os.path.join(args.ego4d_embeddings_path, "shape.json"), "r")
    )
    print(f"Loaded {ego4d_embeddings_shape['shape'][0]} ego4d embeddings")
    ego4d_unique_captions = json.load(
        open(os.path.join(args.ego4d_embeddings_path, "captions.json"), "r")
    )
    print(f"Loaded {len(ego4d_unique_captions)} ego4d unique captions")

    print(f"Opening {args.lavila_embeddings_path} embeddings")
    lavila_embeddings_shape = json.load(
        open(os.path.join(args.lavila_embeddings_path, "shape.json"), "r")
    )
    print(f"Loaded {lavila_embeddings_shape['shape'][0]} lavila embeddings")
    lavila_unique_captions = json.load(
        open(os.path.join(args.lavila_embeddings_path, "captions.json"), "r")
    )
    print(f"Loaded {len(lavila_unique_captions)} lavila unique captions")

    # Group samples by video for batch processing
    video_groups = group_samples_by_video(data)
    print(f"Processing {len(video_groups)} unique videos")

    # Merge containers
    results_dict: Dict[int, Tuple[str, float, float, str]] = {}
    stats_dict = {
        "old_timestamps_duration": [],
        "new_timestamps_duration": [],
        "iou": [],
        "expansion_ratio": [],
        "mean_similarity_between_segments": [],
    }

    # Prepare args for worker initializer (only simple types!)
    init_args = {
        "ego4d_embeddings_path": args.ego4d_embeddings_path,
        "lavila_embeddings_path": args.lavila_embeddings_path,
        "chunk_metadata_root": args.chunk_metadata_root,
        "tau": args.tau,
    }

    items = list(video_groups.items())
    total_videos = len(items)

    if args.num_workers == 1:
        # Serial fallback
        _init_worker(init_args)
        for i, item in enumerate(
            tqdm(items, total=total_videos, desc="Processing videos (serial)")
        ):
            local_results, local_stats = _process_one_video(item, preprocess_captions)
            results_dict.update(local_results)
            for k in stats_dict:
                stats_dict[k].extend(local_stats[k])
            if ((i + 1) % args.log_every) == 0:
                wandb.log({"progress": (i + 1) / total_videos}, step=i + 1)
    else:
        # Parallel path
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(init_args,),
        ) as ex:
            futures = [
                ex.submit(_process_one_video, item, preprocess_captions)
                for item in items
            ]
            for i, fut in enumerate(
                tqdm(
                    as_completed(futures),
                    total=total_videos,
                    desc="Processing videos (parallel)",
                )
            ):
                local_results, local_stats = fut.result()
                results_dict.update(local_results)

                for k in stats_dict:
                    stats_dict[k].extend(local_stats[k])

                if ((i + 1) % args.log_every) == 0:
                    wandb.log({"progress": (i + 1) / total_videos}, step=i + 1)

    results = [results_dict[i] for i in range(len(data))]

    os.makedirs(Path(args.output_path) / args.embedding_model, exist_ok=True)

    output_file = (
        Path(args.output_path)
        / args.embedding_model
        / f"ego4d_train_temperature_.pkl"
    )
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} results to {output_file}")

    shifted_timestamps_distribution = plot_distribution(
        values=stats_dict["new_timestamps_duration"],
        title="Segment Lengths Histogram (Shifted Timestamps)",
    )

    mean_similarity_between_segments_distribution = plot_distribution(
        values=stats_dict["mean_similarity_between_segments"],
        title="Mean Similarity Between Segments Histogram",
        xlabel="Mean Similarity Between Segments",
        ylabel="Value",
    )

    table = wandb.Table(
        columns=[
            "Dataset",
            "Mean Timestamp Duration",
            "Std Timestamp Duration",
            "Mean IoU",
            "Std IoU",
            "Mean Expansion Ratio",
            "Std Expansion Ratio",
        ]
    )

    table.add_data(
        "Shifted",
        np.mean(stats_dict["new_timestamps_duration"]),
        np.std(stats_dict["new_timestamps_duration"]),
        np.mean(stats_dict["iou"]),
        np.std(stats_dict["iou"]),
        np.mean(stats_dict["expansion_ratio"]),
        np.std(stats_dict["expansion_ratio"]),
    )

    wandb.log(
        {
            "table": table,
            "shifted_timestamp_dist": wandb.Image(shifted_timestamps_distribution),
            "mean_similarity_between_segments_dist": wandb.Image(
                mean_similarity_between_segments_distribution
            ),
        }
    )

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
        "--output-path", type=str, required=True, help="Pickle output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-every", type=int, default=100, help="Log progress every N videos"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes for per-video processing",
    )
    parser.add_argument(
        "--preprocess-function",
        type=str,
        default="preprocess_captions",
        help="Function to use to preprocess the captions",
    )
    args = parser.parse_args()
    main(args)
