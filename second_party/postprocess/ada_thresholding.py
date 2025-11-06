import os
import json
import math
import random
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import wandb
from tqdm import tqdm
from matplotlib import pyplot as plt


# ==================== Plotting ====================


def plot_distribution(
    values: List[float],
    title: str,
    xlabel: str = "Length (seconds)",
    ylabel: str = "Frequency",
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==================== I/O & Basic Utils ====================


def load_all_chunks_metadata_for_video(chunk_metadata_root: str, video_id: str):
    """
    Load per-chunk captions metadata for a given video and flatten it.
    Expects files like:
        <chunk_metadata_root>/<video_id>/<chunk_start>.mp4/captions.json
    """
    base_video_path = Path(chunk_metadata_root) / video_id
    segments = []
    for chunk_path in base_video_path.glob("*.mp4/captions.json"):
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_start_offset = int(str(chunk_path).split("/")[-2].split(".")[0])
            metadata = json.load(f)["metadata"]
            for m in metadata:
                m["timestamps"] = [x + chunk_start_offset for x in m["timestamps"]]
            segments.extend(metadata)
    segments.sort(key=lambda x: x["timestamps"][0])
    return segments


def resolve_anchor_index(
    video_id: str, anchor_timestamp: float, flattened_metadata: List[Dict[str, Any]]
) -> int:
    """Find the segment index whose midpoint is closest to the given timestamp."""
    if not flattened_metadata:
        raise ValueError(f"Flattened metadata is empty for video {video_id}")
    best_idx, best_dist = -1, float("inf")
    for idx, m in enumerate(flattened_metadata):
        s, e = m["timestamps"][0], m["timestamps"][-1]
        mid = 0.5 * (s + e)
        d = abs(anchor_timestamp - mid)
        if d < best_dist:
            best_idx, best_dist = idx, d
    if best_idx < 0:
        raise ValueError(
            f"Could not resolve anchor index for {video_id} @ {anchor_timestamp}"
        )
    return best_idx


def group_samples_by_video(data: List[Tuple]) -> Dict[str, List[Tuple]]:
    """
    Group dataset entries by video_id.
    data entries are tuples: (video_id, start, end, caption)
    """
    groups = defaultdict(list)
    for idx, sample in enumerate(data):
        video_id, start, end, original_caption = sample
        groups[video_id].append((idx, start, end, original_caption))
    return groups


def indices_to_times(
    flattened_metadata: List[Dict[str, Any]], start_idx: int, end_idx: int
) -> Tuple[float, float]:
    s_time = math.floor(flattened_metadata[start_idx]["timestamps"][0])
    e_time = math.ceil(flattened_metadata[end_idx]["timestamps"][-1])
    return float(s_time), float(e_time)


def compute_intersection_stats(
    original_start: float, original_end: float, new_start: float, new_end: float
) -> Dict[str, float]:
    inter_s, inter_e = max(original_start, new_start), min(original_end, new_end)
    inter = max(0.0, inter_e - inter_s)
    union = max(original_end, new_end) - min(original_start, new_start)
    iou = inter / union if union > 0 else 0.0
    orig_dur = max(1e-6, original_end - original_start)
    new_dur = max(0.0, new_end - new_start)
    return {"iou": iou, "expansion_ratio": new_dur / orig_dur}


# ==================== Query-Dependent Scoring ====================


def build_segment_caption_row_indices(
    flattened_metadata: List[Dict[str, Any]],
    lavila_unique_captions: Dict[str, int],
    preprocess_captions: Callable,
) -> List[np.ndarray]:
    """
    For each segment, map its captions to row indices in lavila_embeddings.
    Returns a list of np.ndarray[int] (one per segment).
    """
    seg_rows: List[np.ndarray] = []
    for seg in flattened_metadata:
        rows = []
        for caption in seg.get("captions", []):
            processed = preprocess_captions([caption])[0]
            idx = lavila_unique_captions.get(processed, None)
            if idx is not None:
                rows.append(idx)
        seg_rows.append(np.asarray(rows, dtype=np.int64))
    return seg_rows


def compute_similarity_curve_max_over_captions(
    anchor_vec: np.ndarray,
    lavila_embeddings: np.ndarray,
    seg_rows: List[np.ndarray],
    return_best_rows: bool = False,
) -> Any:
    """
    Build S where S[i] is the max cosine similarity between the anchor
    and any caption embedding for segment i. Assumes embeddings are L2-normalized.
    If return_best_rows, also returns array of best row indices per segment (or -1).
    """
    S = np.empty(len(seg_rows), dtype=np.float32)
    if return_best_rows:
        best_rows = np.full(len(seg_rows), -1, dtype=np.int64)

    for i, rows in enumerate(seg_rows):
        if rows.size == 0:
            S[i] = -1.0
        else:
            E = lavila_embeddings[rows]  # (m, d)
            sims = E @ anchor_vec  # (m,)
            j = int(np.argmax(sims))
            S[i] = float(sims[j])
            if return_best_rows:
                best_rows[i] = rows[j]

    return (S, best_rows) if return_best_rows else S


# ==================== Adaptive Thresholding Span Generator ====================


def adaptive_threshold(scores: np.ndarray, eta_bins: int, kappa_count: int) -> float:
    """
    Compute adaptive threshold γ via inverse cumulative histogram over scores
    using η bins. Traverse from high to low and pick the first bin whose
    cumulative count >= κ; return that bin's left edge as γ.
    """
    if len(scores) == 0:
        return 1.0
    counts, edges = np.histogram(scores, bins=eta_bins)
    cum_rev = np.cumsum(counts[::-1])
    pos = np.searchsorted(cum_rev, kappa_count, side="left")
    if pos >= len(counts):
        pos = len(counts) - 1
    orig_idx = len(counts) - 1 - pos
    gamma = float(edges[orig_idx])
    return gamma


def span_generator(
    scores: np.ndarray, gamma: float, tau_consec: int
) -> List[Tuple[int, int]]:
    """
    Generate candidate spans by scanning scores over time:
      - Start when score >= γ
      - End when τ consecutive scores fall below γ
    Returns list of (start_idx, end_idx) inclusive.
    """
    spans = []
    active = False
    start = None
    below = 0

    for i, s in enumerate(scores):
        if not active:
            if s >= gamma:
                active = True
                start = i
                below = 0
        else:
            if s < gamma:
                below += 1
                if below >= tau_consec:
                    end = i - tau_consec  # last index >= γ before the τ run
                    if end >= start:
                        spans.append((start, end))
                    active = False
                    start = None
                    below = 0
            else:
                below = 0

    if active:
        spans.append((start, len(scores) - 1))
    return spans


def choose_span_around_anchor(spans: List[Tuple[int, int]], anchor_idx: int) -> Any:
    """Pick the span that contains anchor_idx; fallback to the one whose center is closest."""
    if not spans:
        return None
    for s, e in spans:
        if s <= anchor_idx <= e:
            return (s, e)
    centers = [0.5 * (s + e) for s, e in spans]
    best = min(range(len(spans)), key=lambda i: abs(centers[i] - anchor_idx))
    return spans[best]


# ==================== Parallel Worker ====================

_WORKER: Dict[str, Any] = {
    "args": None,
    "ego4d_embeddings": None,
    "ego4d_unique_captions": None,
    "lavila_embeddings": None,
    "lavila_unique_captions": None,
}


def _init_worker(args_dict: Dict[str, Any]):
    _WORKER["args"] = args_dict

    # Ego4D (anchor caption features)
    ego_shape = json.load(
        open(os.path.join(args_dict["ego4d_embeddings_path"], "shape.json"), "r")
    )
    _WORKER["ego4d_embeddings"] = np.memmap(
        os.path.join(args_dict["ego4d_embeddings_path"], "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(ego_shape["shape"]),
    )
    _WORKER["ego4d_unique_captions"] = json.load(
        open(os.path.join(args_dict["ego4d_embeddings_path"], "captions.json"), "r")
    )

    # LaViLa (segment caption features)
    lav_shape = json.load(
        open(os.path.join(args_dict["lavila_embeddings_path"], "shape.json"), "r")
    )
    _WORKER["lavila_embeddings"] = np.memmap(
        os.path.join(args_dict["lavila_embeddings_path"], "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(lav_shape["shape"]),
    )
    _WORKER["lavila_unique_captions"] = json.load(
        open(os.path.join(args_dict["lavila_embeddings_path"], "captions.json"), "r")
    )


def _process_one_video(payload: Tuple[str, List[Tuple]], preprocess_captions: Callable):
    """
    For one video: build per-segment caption indices once, then for each sample:
      - get anchor embedding
      - compute S via max-over-captions per segment (+ track best row per seg)
      - adaptive threshold -> spans -> pick span around anchor
      - compute intra-span mean similarity between the chosen caption embeddings
    """
    video_id, samples = payload
    args = _WORKER["args"]

    # 1) Flattened metadata
    flattened = load_all_chunks_metadata_for_video(
        args["chunk_metadata_root"], f"{video_id}.mp4"
    )

    # 2) Pre-index LaViLa caption rows (reused for all anchors in this video)
    seg_rows = build_segment_caption_row_indices(
        flattened_metadata=flattened,
        lavila_unique_captions=_WORKER["lavila_unique_captions"],
        preprocess_captions=preprocess_captions,
    )

    ego_emb = _WORKER["ego4d_embeddings"]
    ego_caps = _WORKER["ego4d_unique_captions"]
    lavila_emb = _WORKER["lavila_embeddings"]

    local_results: Dict[int, Tuple[str, float, float, str]] = {}
    local_stats = {
        "old_timestamps_duration": [],
        "new_timestamps_duration": [],
        "iou": [],
        "expansion_ratio": [],
        "mean_similarity_between_segments": [],
    }

    for original_idx, start, end, original_caption in samples:
        try:
            anchor_ts = 0.5 * (start + end)
            anchor_idx = resolve_anchor_index(video_id, anchor_ts, flattened)

            # Anchor embedding (Ego4D); normalize defensively
            cap = preprocess_captions([original_caption])[0]
            a = ego_emb[ego_caps[cap]]
            n = np.linalg.norm(a)
            anchor_vec = a / n if n > 0 else a

            # Query-dependent similarity curve S (+ which caption row was best per segment)
            S, best_rows = compute_similarity_curve_max_over_captions(
                anchor_vec=anchor_vec,
                lavila_embeddings=lavila_emb,
                seg_rows=seg_rows,
                return_best_rows=True,
            )

            # Adaptive thresholding -> spans -> pick the one around the anchor
            gamma = adaptive_threshold(S, args["eta_bins"], args["kappa_count"])
            spans = span_generator(S, gamma, args["tau_consecutive"])
            picked = choose_span_around_anchor(spans, anchor_idx)

            if picked is None:
                new_start, new_end = float(start), float(end)
                rows_in_span = []
            else:
                idx_s, idx_e = picked
                new_start, new_end = indices_to_times(flattened, idx_s, idx_e)
                # Best caption rows used inside the span (for intra-span similarity)
                rows_in_span = [
                    best_rows[i] for i in range(idx_s, idx_e + 1) if best_rows[i] != -1
                ]

            # Intra-span mean similarity between segments (off-diagonal mean)
            if len(rows_in_span) >= 2:
                E = lavila_emb[rows_in_span]
                # Normalize defensively (embeddings are expected normalized already)
                norms = np.linalg.norm(E, axis=1, keepdims=True)
                E = E / np.where(norms > 0, norms, 1.0)
                M = E @ E.T
                mask = ~np.eye(M.shape[0], dtype=bool)
                mean_sim = float(np.mean(M[mask]))
                local_stats["mean_similarity_between_segments"].append(mean_sim)

                if mean_sim < args["keep_threshold"]:
                    new_start, new_end = float(start), float(end)
                    rows_in_span = []

        except Exception as e:
            print(f"[{video_id}] Fallback due to error: {e}")
            new_start, new_end = float(start), float(end)

        local_results[original_idx] = (video_id, new_start, new_end, original_caption)

        # Metrics
        local_stats["old_timestamps_duration"].append(end - start)
        local_stats["new_timestamps_duration"].append(new_end - new_start)
        s = compute_intersection_stats(start, end, new_start, new_end)
        local_stats["iou"].append(s["iou"])
        local_stats["expansion_ratio"].append(s["expansion_ratio"])

    return local_results, local_stats


# ==================== Main ====================


def main(args):
    # Choose preprocess function
    if args.preprocess_function == "preprocess_captions":
        from second_party.preprocess.utils import preprocess_captions
    elif args.preprocess_function == "preprocess_caption_v2":
        from second_party.preprocess.utils import (
            preprocess_caption_v2 as preprocess_captions,
        )
    else:
        raise ValueError(f"Invalid preprocess function: {args.preprocess_function}")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset of (video_id, start, end, caption)
    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    # (Optional) peek at shapes/maps for info logs
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

    # Group samples by video
    video_groups = group_samples_by_video(data)
    print(f"Processing {len(video_groups)} unique videos")

    # ---- Weights & Biases ----
    wandb_run_name = (
        args.wandb_run_name
        or f"QDep eta{args.eta_bins}-k{args.kappa_count}-tau{args.tau_consecutive}-keep{args.keep_threshold}"
    )
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config={
            "eta_bins": args.eta_bins,
            "kappa_count": args.kappa_count,
            "tau_consecutive": args.tau_consecutive,
            "preprocess_function": args.preprocess_function,
            "seed": args.seed,
            "num_workers": args.num_workers,
        },
        group=args.wandb_group or "Adaptive Thresholding - QueryDependent",
    )

    # Worker init params
    init_args = {
        "ego4d_embeddings_path": args.ego4d_embeddings_path,
        "lavila_embeddings_path": args.lavila_embeddings_path,
        "chunk_metadata_root": args.chunk_metadata_root,
        "eta_bins": args.eta_bins,
        "kappa_count": args.kappa_count,
        "tau_consecutive": args.tau_consecutive,
    }

    results_dict: Dict[int, Tuple[str, float, float, str]] = {}
    stats = {
        "old_timestamps_duration": [],
        "new_timestamps_duration": [],
        "iou": [],
        "expansion_ratio": [],
        "mean_similarity_between_segments": [],
    }

    items = list(video_groups.items())
    total_videos = len(items)

    if args.num_workers == 1:
        _init_worker(init_args)
        for i, item in enumerate(
            tqdm(items, total=total_videos, desc="Processing videos (serial)")
        ):
            local_results, local_stats = _process_one_video(item, preprocess_captions)
            results_dict.update(local_results)
            for k in stats:
                stats[k].extend(local_stats[k])
            if ((i + 1) % args.log_every) == 0:
                wandb.log({"progress": (i + 1) / total_videos}, step=i + 1)
    else:
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
                for k in stats:
                    stats[k].extend(local_stats[k])
                if ((i + 1) % args.log_every) == 0:
                    wandb.log({"progress": (i + 1) / total_videos}, step=i + 1)

    # Reconstruct in original order
    results = [results_dict[i] for i in range(len(data))]

    # ---- Metrics & Plots ----
    # Basic means for printing
    iou_mean = float(np.mean(stats["iou"])) if stats["iou"] else -1.0
    er_mean = (
        float(np.mean(stats["expansion_ratio"])) if stats["expansion_ratio"] else -1.0
    )
    print(f"Mean IoU: {iou_mean:.4f} | Mean Expansion Ratio: {er_mean:.4f}")

    # Distributions
    original_distribution = plot_distribution(
        values=stats["old_timestamps_duration"],
        title="Segment Lengths Histogram (Original)",
    )
    shifted_distribution = plot_distribution(
        values=stats["new_timestamps_duration"],
        title="Segment Lengths Histogram (Shifted Timestamps)",
    )
    mean_sim_distribution = plot_distribution(
        values=stats["mean_similarity_between_segments"],
        title="Mean Similarity Between Segments (Chosen Captions per Span)",
        xlabel="Mean Similarity Between Segments",
        ylabel="Value",
    )

    # Table summary (match your original schema)
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
        (
            float(np.mean(stats["old_timestamps_duration"]))
            if stats["old_timestamps_duration"]
            else -1.0
        ),
        (
            float(np.std(stats["old_timestamps_duration"]))
            if stats["old_timestamps_duration"]
            else -1.0
        ),
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    )
    table.add_data(
        "Shifted",
        (
            float(np.mean(stats["new_timestamps_duration"]))
            if stats["new_timestamps_duration"]
            else -1.0
        ),
        (
            float(np.std(stats["new_timestamps_duration"]))
            if stats["new_timestamps_duration"]
            else -1.0
        ),
        float(np.mean(stats["iou"])) if stats["iou"] else -1.0,
        float(np.std(stats["iou"])) if stats["iou"] else -1.0,
        float(np.mean(stats["expansion_ratio"])) if stats["expansion_ratio"] else -1.0,
        float(np.std(stats["expansion_ratio"])) if stats["expansion_ratio"] else -1.0,
    )
    
    wandb.log(
        {
            "table": table,
            "original_timestamp_dist": wandb.Image(original_distribution),
            "shifted_timestamp_dist": wandb.Image(shifted_distribution),
            "mean_similarity_between_segments_dist": wandb.Image(mean_sim_distribution),
        }
    )

    try:
        wandb.log(
            {
                "max_segment_length": np.max(stats["new_timestamps_duration"]),
                "min_segment_length": np.min(stats["new_timestamps_duration"]),
            }
        )
    except Exception as e:
        print(f"Error logging to W&B: {e}")

    # # Persist outputs
    # os.makedirs(args.output_path, exist_ok=True)
    # out_file = (
    #     Path(args.output_path)
    #     / args.embedding_model
    #     / f"ego4d_train_ada_thresholding_eta{args.eta_bins}_k{args.kappa_count}_tau{args.tau_consecutive}_keep{args.keep_threshold}.pkl"
    # )

    # os.makedirs(out_file, exist_ok=True)

    # with open(out_file, "wb") as f:
    #     pickle.dump(results, f)
    # print(f"Saved {len(results)} results to {out_file}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timestamp shifting via query-dependent scoring (max-over-captions) + adaptive threshold spans, with W&B logging."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Pickle with (video_id, start, end, caption)",
    )
    parser.add_argument(
        "--ego4d-embeddings-path",
        type=str,
        required=True,
        help="Path with Ego4D normalized caption embeddings",
    )
    parser.add_argument(
        "--lavila-embeddings-path",
        type=str,
        required=True,
        help="Path with LaViLa normalized caption embeddings",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Embedding model used to extract embeddings",
    )
    parser.add_argument(
        "--chunk-metadata-root",
        type=str,
        required=True,
        help="Root with per-chunk captions metadata JSONs",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Folder to write output .pkl"
    )
    parser.add_argument(
        "--preprocess-function",
        type=str,
        default="preprocess_captions",
        help="One of {preprocess_captions, preprocess_caption_v2}",
    )
    # Span generator hyper-params
    parser.add_argument(
        "--eta-bins",
        type=int,
        default=10,
        help="η: histogram bins for adaptive threshold",
    )
    parser.add_argument(
        "--kappa-count", type=int, default=7, help="κ: inverse-cumulative count cutoff"
    )
    parser.add_argument(
        "--tau-consecutive",
        type=int,
        default=5,
        help="τ: consecutive below-γ to end a span",
    )
    # W&B + runtime
    parser.add_argument("--wandb-project", type=str, default="Thesis")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--log-every", type=int, default=100, help="Log progress every N videos"
    )
    parser.add_argument("--keep-threshold", type=float, default=0.6)
    args = parser.parse_args()
    main(args)
