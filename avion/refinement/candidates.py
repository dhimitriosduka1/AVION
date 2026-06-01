import math

from avion.refinement.io import as_float, public_record, record_key


def round_time(value, resolution):
    if resolution <= 0:
        return float(value)
    rounded = round(round(float(value) / resolution) * resolution, 6)
    return 0.0 if abs(rounded) < 1e-9 else rounded


def anchor_distance(start, end, timestamp):
    if timestamp is None:
        return 0.0
    if start <= timestamp <= end:
        return 0.0
    return min(abs(timestamp - start), abs(timestamp - end))


def candidate_distance(candidate, record):
    start = as_float(candidate["start_sec"])
    end = as_float(candidate["end_sec"])
    qwen_start = as_float(record.get("qwen_start_sec"), start)
    qwen_end = as_float(record.get("qwen_end_sec"), end)
    timestamp = as_float(record.get("timestamp_sec"), 0.5 * (qwen_start + qwen_end))
    center = 0.5 * (start + end)
    return abs(start - qwen_start) + abs(end - qwen_end) + 0.1 * abs(center - timestamp)


def temporal_iou(a, b):
    if a is None or b is None:
        return 0.0
    start_a, end_a = a
    start_b, end_b = b
    if start_a >= end_a or start_b >= end_b:
        return 0.0
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return inter / union


def _add_candidate(raw_candidates, start, end, source):
    if start is None or end is None:
        return
    if not math.isfinite(start) or not math.isfinite(end):
        return
    raw_candidates.append((float(start), float(end), source))


def _neighbor_aware_candidates(record, raw_candidates):
    timestamp = as_float(record.get("timestamp_sec"))
    prev_timestamp = as_float(record.get("previous_timestamp_sec"))
    next_timestamp = as_float(record.get("next_timestamp_sec"))
    qwen_start = as_float(record.get("qwen_start_sec"))
    qwen_end = as_float(record.get("qwen_end_sec"))
    if timestamp is None or prev_timestamp is None or next_timestamp is None:
        return

    left_limit = 0.5 * (prev_timestamp + timestamp)
    right_limit = 0.5 * (timestamp + next_timestamp)
    _add_candidate(raw_candidates, left_limit, right_limit, "neighbor_midpoints")
    for pad in (0.25, 0.5):
        _add_candidate(
            raw_candidates,
            left_limit + pad,
            right_limit - pad,
            f"neighbor_midpoints_inner_{pad:g}",
        )
        _add_candidate(
            raw_candidates,
            left_limit - pad,
            right_limit + pad,
            f"neighbor_midpoints_outer_{pad:g}",
        )
    if qwen_start is not None and qwen_end is not None:
        _add_candidate(raw_candidates, left_limit, qwen_end, "neighbor_left_qwen_end")
        _add_candidate(raw_candidates, qwen_start, right_limit, "qwen_start_neighbor_right")
        _add_candidate(
            raw_candidates,
            max(left_limit, qwen_start),
            min(right_limit, qwen_end),
            "qwen_clipped_to_neighbor_midpoints",
        )


def generate_candidate_windows(record, config):
    cfg = config["candidate_generation"]
    qwen_start = as_float(record.get("qwen_start_sec"))
    qwen_end = as_float(record.get("qwen_end_sec"))
    timestamp = as_float(record.get("timestamp_sec"))
    video_duration = as_float(record.get("video_duration_sec"))
    errors = []

    if qwen_start is None or qwen_end is None or qwen_start >= qwen_end:
        return [], ["invalid_qwen_segment"]
    if timestamp is None:
        timestamp = 0.5 * (qwen_start + qwen_end)
    if video_duration is not None and video_duration <= 0:
        errors.append("invalid_video_duration")
        video_duration = None

    raw_candidates = []
    _add_candidate(raw_candidates, qwen_start, qwen_end, "qwen_original")

    for start_shift in cfg["start_shifts"]:
        for end_shift in cfg["end_shifts"]:
            _add_candidate(
                raw_candidates,
                qwen_start + float(start_shift),
                qwen_end + float(end_shift),
                "boundary_shift",
            )

    qwen_duration = qwen_end - qwen_start
    qwen_center = 0.5 * (qwen_start + qwen_end)
    for scale in cfg["duration_scales"]:
        duration = qwen_duration * float(scale)
        _add_candidate(
            raw_candidates,
            qwen_center - 0.5 * duration,
            qwen_center + 0.5 * duration,
            "duration_scale",
        )

    for duration in cfg["timestamp_centered_durations"]:
        duration = float(duration)
        _add_candidate(
            raw_candidates,
            timestamp - 0.5 * duration,
            timestamp + 0.5 * duration,
            "timestamp_centered",
        )

    _neighbor_aware_candidates(record, raw_candidates)

    candidates_by_key = {}
    for start, end, source in raw_candidates:
        if cfg.get("clip_to_video_bounds", True):
            start = max(0.0, start)
            if video_duration is not None:
                end = min(video_duration, end)
        start = round_time(start, cfg["time_resolution"])
        end = round_time(end, cfg["time_resolution"])

        if start >= end:
            continue
        duration = end - start
        if duration < cfg["min_duration"] or duration > cfg["max_duration"]:
            continue
        if start < 0:
            continue
        if video_duration is not None and end > video_duration:
            continue
        if anchor_distance(start, end, timestamp) > cfg["anchor_tolerance"]:
            continue

        prev_timestamp = as_float(record.get("previous_timestamp_sec"))
        next_timestamp = as_float(record.get("next_timestamp_sec"))
        max_crossing = cfg["max_neighbor_crossing"]
        if prev_timestamp is not None and start < prev_timestamp - max_crossing:
            continue
        if next_timestamp is not None and end > next_timestamp + max_crossing:
            continue

        key = (start, end)
        if key not in candidates_by_key:
            candidates_by_key[key] = {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 6),
                "center_sec": round(0.5 * (start + end), 6),
                "sources": [source],
            }
        elif source not in candidates_by_key[key]["sources"]:
            candidates_by_key[key]["sources"].append(source)

    candidates = list(candidates_by_key.values())
    candidates.sort(key=lambda candidate: candidate_distance(candidate, record))

    max_candidates = cfg.get("max_candidates_per_narration")
    if max_candidates and len(candidates) > max_candidates:
        kept = candidates[: int(max_candidates)]
        qwen_segment = (round_time(qwen_start, cfg["time_resolution"]), round_time(qwen_end, cfg["time_resolution"]))
        qwen_candidate = next(
            (
                candidate
                for candidate in candidates
                if (candidate["start_sec"], candidate["end_sec"]) == qwen_segment
            ),
            None,
        )
        if qwen_candidate is not None and qwen_candidate not in kept:
            kept[-1] = qwen_candidate
        candidates = kept

    candidates.sort(key=lambda candidate: (candidate["start_sec"], candidate["end_sec"]))
    for idx, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"c{idx:03d}"

    if not candidates:
        errors.append("too_few_valid_candidates")
    return candidates, errors


def build_candidate_record(record, config):
    candidates, errors = generate_candidate_windows(record, config)
    return {
        "record_index": record.get("_record_index"),
        "narration_uid": record.get("narration_uid"),
        "video_uid": record.get("video_uid"),
        "clip_uid": record.get("clip_uid"),
        "timestamp_sec": record.get("timestamp_sec"),
        "text": record.get("text"),
        "qwen_start_sec": record.get("qwen_start_sec"),
        "qwen_end_sec": record.get("qwen_end_sec"),
        "video_duration_sec": record.get("video_duration_sec"),
        "previous_narration_uid": record.get("previous_narration_uid"),
        "previous_timestamp_sec": record.get("previous_timestamp_sec"),
        "previous_text": record.get("previous_text"),
        "next_narration_uid": record.get("next_narration_uid"),
        "next_timestamp_sec": record.get("next_timestamp_sec"),
        "next_text": record.get("next_text"),
        "record_key": record_key(record),
        "record": public_record(record),
        "candidates": candidates,
        "errors": errors,
    }

