import pickle
import os
import random
import argparse
import numpy as np
import decord
import wandb
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm


def plot_segment_len_dist(segment_lengths: List[float], title: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.hist(segment_lengths, bins=100)
    ax.set_title(title)
    ax.set_xlabel("Length (seconds)")
    ax.set_ylabel("Frequency")
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
    Returns IoU, expansion ratio, and absolute change in duration.
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


def jitter_scale_window(
    start: float,
    end: float,
    min_duration: float = 1.0,
    max_duration: float = 5.0,
    min_start: float = 0.0,
    video_duration: float = 1.0,
    scale_factor: float = 1.0,
):
    """
    Keep center fixed. Grow/shrink window by a random scale s.
    Enforce duration bounds. If window goes out of bounds,
    shift it to stay within [min_start, video_duration]
    while preserving the new duration.
    """
    c = 0.5 * (start + end)
    d = max(end - start, 1e-6)  # avoid zero

    new_d = max(min(min(d * scale_factor, max_duration), video_duration), min_duration)

    new_start = c - new_d / 2.0
    new_end = c + new_d / 2.0

    if new_end > video_duration:
        excess = new_end - video_duration
        new_start -= excess
        new_end = video_duration

    if new_start < min_start:
        # Shift right
        excess = min_start - new_start
        new_start = min_start
        new_end += excess

        if new_end > video_duration:
            new_end = video_duration

    return new_start, new_end


def _probe_duration(video_root: Path, vid: str) -> tuple[str, Optional[float]]:
    path = video_root / f"{vid}.mp4"
    if not path.exists():
        return vid, None
    try:
        vr = decord.VideoReader(str(path))
        fps = max(float(vr.get_avg_fps()), 1e-6)
        return vid, (len(vr) / fps)
    except Exception:
        return vid, None


def index_video_durations(
    video_root: Path, video_ids: List[str]
) -> Dict[str, Optional[float]]:
    """
    Multithreaded cache of video_id -> duration_seconds.
    Waits for all threads to complete before returning.
    """
    durations: Dict[str, Optional[float]] = {}
    unique_ids = sorted(set(video_ids))
    # For I/O-bound tasks, many threads are OK:
    workers = min(32, (os.cpu_count() or 1) * 5)
    print(f"Using {workers} workers")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_probe_duration, video_root, vid): vid for vid in unique_ids
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Indexing video metadata"
        ):
            vid, dur = fut.result()
            durations[vid] = dur

    return durations


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"

    # Reproducibility
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    wandb.init(
        project="Thesis",
        name=f"Random Shift - {args.scale_min} - {args.scale_max} - {args.min_duration}",
        config={**args.__dict__},
        group=f"Random Timestamp Shift",
    )

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
        total = len(data)
    print(f"Loaded {total} samples")

    if len(data) == 0:
        print("Dataset is empty; exiting early.")
        wandb.finish()
        return

    # Pre-index durations once
    video_ids = [sample[0] for sample in data]
    durations = index_video_durations(Path(args.video_root), video_ids)

    max_duration = durations[max(durations, key=durations.get)]

    scales = rng.uniform(args.scale_min, args.scale_max, size=total)

    result = {
        "old_timestamps_duration": [],
        "new_timestamps_duration": [],
        "new_data": [],
        "iou": [],
        "expansion_ratio": [],
    }

    for i, sample in tqdm(enumerate(data), desc="Jittering timestamps"):
        vid, start, end, meta = sample
        vdur = durations.get(vid, None)

        new_start, new_end = jitter_scale_window(
            start=sample[1],
            end=sample[2],
            min_duration=args.min_duration,
            max_duration=max_duration,
            video_duration=vdur,
            scale_factor=float(scales[i]),
        )

        result["old_timestamps_duration"].append(end - start)
        result["new_timestamps_duration"].append(new_end - new_start)

        compute_stats = compute_intersection_stats(start, end, new_start, new_end)
        result["iou"].append(compute_stats["iou"])
        result["expansion_ratio"].append(compute_stats["expansion_ratio"])

        result["new_data"].append((vid, new_start, new_end, meta))

        # Throttle W&B logging
        if (i % args.log_every) == 0:
            wandb.log({"progress": (i + 1) / total}, step=i + 1)

    with open(
        os.path.join(
            Path(args.output_path) / "random_shift_timestamps",
            f"ego4d_train_random_shift_{args.scale_min}_{args.scale_max}_{args.min_duration}_{args.seed}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(result["new_data"], f)

    original_distribution = plot_segment_len_dist(
        segment_lengths=result["old_timestamps_duration"],
        title="Segment Lengths Histogram (Original)",
    )

    shifted_timestamps_distribution = plot_segment_len_dist(
        segment_lengths=result["new_timestamps_duration"],
        title="Segment Lengths Histogram (Shifted Timestamps)",
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
        "Original",
        np.mean(result["old_timestamps_duration"]),
        np.std(result["old_timestamps_duration"]),
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    )

    table.add_data(
        "Shifted",
        np.mean(result["new_timestamps_duration"]),
        np.std(result["new_timestamps_duration"]),
        np.mean(result["iou"]),
        np.std(result["iou"]),
        np.mean(result["expansion_ratio"]),
        np.std(result["expansion_ratio"]),
    )

    wandb.log(
        {
            "table": table,
            "original_timestamp_dist": wandb.Image(original_distribution),
            "shifted_timestamp_dist": wandb.Image(shifted_timestamps_distribution),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=20.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()

    main(args)
