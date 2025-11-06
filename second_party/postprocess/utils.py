import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from matplotlib import pyplot as plt

import os


def plot_distribution(
    values: List[float],
    title: str,
    xlabel: str = "Length (seconds)",
    ylabel: str = "Frequency",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.hist(values, bins=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def compute_intersection_stats(
    original_start: float,
    original_end: float,
    new_start: float,
    new_end: float,
) -> Dict[str, float]:
    """
    Compute intersection statistics between original and new time windows.
    Returns IoU and expansion ratio.
    """
    # Intersection
    intersection_start = max(original_start, new_start)
    intersection_end = min(original_end, new_end)
    intersection = max(0, intersection_end - intersection_start)

    # Union
    union_start = min(original_start, new_start)
    union_end = max(original_end, new_end)
    union = union_end - union_start

    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0

    # Duration metrics
    original_duration = original_end - original_start
    new_duration = new_end - new_start
    expansion_ratio = new_duration / original_duration if original_duration > 0 else 1.0

    return {
        "iou": iou,
        "expansion_ratio": expansion_ratio,
    }


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
            # Update the timestamp so that it is offset by the chunk start
            for m in metadata:
                m["timestamps"] = [x + chunk_start_offset for x in m["timestamps"]]
            captions.extend(metadata)
    # Sort by start timestamp
    captions.sort(key=lambda x: x["timestamps"][0])
    return captions


def resolve_anchor_index(
    video_id: str, anchor_timestamp: float, flattened_metadata: Any
) -> int:
    """
    Resolve the index of the anchor timestamp by finding the closest
    segment midpoint in the flattened metadata.
    """
    if not flattened_metadata:
        raise ValueError(f"Flattened metadata is empty for video {video_id}")

    closest_idx = -1
    min_distance = float("inf")

    for idx, metadata in enumerate(flattened_metadata):
        seg_start = metadata["timestamps"][0]
        seg_end = metadata["timestamps"][-1]
        seg_midpoint = (seg_start + seg_end) * 0.5

        distance = abs(anchor_timestamp - seg_midpoint)

        if distance < min_distance:
            min_distance = distance
            closest_idx = idx

    if closest_idx == -1:
        raise ValueError(
            f"Could not resolve anchor index for video {video_id} with anchor timestamp {anchor_timestamp}"
        )

    return closest_idx


def group_samples_by_video(
    data: List[Tuple],
) -> Dict[str, List[Tuple]]:
    """
    Group samples by video_id for batch processing.
    Returns: video_groups mapping to lists of (idx, start, end, original_caption)
    """
    video_groups = defaultdict(list)
    for idx, sample in enumerate(data):
        video_id, start, end, original_caption = sample
        video_groups[video_id].append((idx, start, end, original_caption))
    return video_groups


_WORKER: Dict[str, Any] = {
    "args": None,
    "ego4d_embeddings": None,
    "ego4d_unique_captions": None,
    "lavila_embeddings": None,
    "lavila_unique_captions": None,
}


def _init_worker(args_dict: Dict[str, Any]):
    """
    Runs once per worker process. Opens memmaps and JSON maps in *this* process.
    """
    _WORKER["args"] = args_dict

    # Ego4D
    ego4d_shape = json.load(
        open(os.path.join(args_dict["ego4d_embeddings_path"], "shape.json"), "r")
    )
    _WORKER["ego4d_embeddings"] = np.memmap(
        os.path.join(args_dict["ego4d_embeddings_path"], "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(ego4d_shape["shape"]),
    )
    _WORKER["ego4d_unique_captions"] = json.load(
        open(os.path.join(args_dict["ego4d_embeddings_path"], "captions.json"), "r")
    )

    # LaViLa
    lavila_shape = json.load(
        open(os.path.join(args_dict["lavila_embeddings_path"], "shape.json"), "r")
    )
    _WORKER["lavila_embeddings"] = np.memmap(
        os.path.join(args_dict["lavila_embeddings_path"], "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(lavila_shape["shape"]),
    )
    _WORKER["lavila_unique_captions"] = json.load(
        open(os.path.join(args_dict["lavila_embeddings_path"], "captions.json"), "r")
    )
