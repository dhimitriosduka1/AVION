import copy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "mode": "lavila_egovlp_ensemble",
    "refinement_iteration": 1,
    "paths": {
        "video_root": None,
        "video_durations": None,
    },
    "candidate_generation": {
        "min_duration": 0.4,
        "max_duration": 8.0,
        "anchor_tolerance": 1.0,
        "max_neighbor_crossing": 1.5,
        "max_candidates_per_narration": 128,
        "time_resolution": 0.1,
        "start_shifts": [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5],
        "end_shifts": [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5],
        "duration_scales": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "timestamp_centered_durations": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        "clip_to_video_bounds": True,
    },
    "scoring": {
        "device": "cuda",
        "batch_size": 16,
        "inference_precision": "fp16",
        "decode_threads": 1,
        "video_chunk_length": 15,
        "video_fps": 30,
        "lavila_clip_length": None,
        "lavila_clip_stride": 16,
        "lavila_norm_style": "openai",
        "egovlp_clip_length": None,
        "egovlp_norm_style": "openai",
    },
    "ranking": {
        "neighbor_weight": 0.5,
        "length_weight": 0.05,
        "qwen_prior_weight": 0.05,
        "anchor_weight": 0.10,
        "preferred_max_duration": 4.0,
        "lavila_weight": 0.5,
        "egovlp_weight": 0.5,
        "agreement_bonus_weight": 0.1,
        "temperature": 0.07,
        "min_score_margin_to_update": 0.05,
        "keep_qwen_if_uncertain": True,
    },
    "confidence": {
        "margin_weight": 1.0,
        "neighbor_weight": 0.5,
        "agreement_weight": 0.5,
        "entropy_weight": 0.5,
        "start_uncertainty_weight": 0.2,
        "end_uncertainty_weight": 0.2,
    },
    "output": {
        "top_k_candidates": 10,
        "store_candidate_rankings": True,
    },
}


def deep_update(base, update):
    result = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None, overrides=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        config = deep_update(config, loaded)
    if overrides:
        config = deep_update(config, overrides)
    return config


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
