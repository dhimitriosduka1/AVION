import json
import math
from pathlib import Path

import numpy as np

from avion.refinement.io import as_float, record_key
from avion.refinement.ranking import MODEL_NAMES, finite_float


def summary_stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def recall_by_iou_threshold(ious, thresholds):
    valid_ious = [float(value) for value in ious if value is not None and math.isfinite(float(value))]
    total = len(valid_ious)
    recalls = {}
    for threshold in thresholds:
        threshold = float(threshold)
        key = f"r@{threshold:g}"
        hits = sum(value >= threshold for value in valid_ious)
        recalls[key] = {
            "threshold": threshold,
            "count": int(hits),
            "total": int(total),
            "recall": float(hits / total) if total else None,
        }
    return recalls


def find_nearest_candidate(candidates, start, end):
    if start is None or end is None or not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: abs(candidate["start_sec"] - start) + abs(candidate["end_sec"] - end),
    )


def candidate_neighbor_success(candidate):
    separations = []
    for model_name in MODEL_NAMES:
        current = finite_float(candidate.get(f"{model_name}_sim_current"))
        if current is None:
            continue
        neighbors = [
            value
            for value in (
                finite_float(candidate.get(f"{model_name}_sim_prev")),
                finite_float(candidate.get(f"{model_name}_sim_next")),
            )
            if value is not None
        ]
        if not neighbors:
            continue
        separations.append(current - max(neighbors))
    if not separations:
        return None
    return sum(separations) / len(separations) > 0


def neighbor_discrimination_accuracy(records, score_records, start_key, end_key):
    score_by_key = {str(record.get("record_key") or record.get("narration_uid")): record for record in score_records}
    successes = []
    for record in records:
        key = str(record_key(record))
        score_record = score_by_key.get(key)
        if score_record is None:
            continue
        candidate = find_nearest_candidate(
            score_record.get("candidates") or [],
            as_float(record.get(start_key)),
            as_float(record.get(end_key)),
        )
        if candidate is None:
            continue
        success = candidate_neighbor_success(candidate)
        if success is not None:
            successes.append(success)
    if not successes:
        return {"count": 0, "accuracy": None}
    return {
        "count": len(successes),
        "accuracy": float(sum(successes) / len(successes)),
    }


def compute_diagnostics(qwen_records, refined_records, score_records=None, time_epsilon=1e-6):
    qwen_by_key = {record_key(record): record for record in qwen_records}
    processed = 0
    skipped = 0
    changed = 0
    start_shifts = []
    end_shifts = []
    before_durations = []
    after_durations = []
    confidence = []
    score_margins = []
    neighbor_separations = []
    agreement = []
    start_uncertainty = []
    end_uncertainty = []

    for refined in refined_records:
        key = record_key(refined)
        qwen = qwen_by_key.get(key, refined)
        qwen_start = as_float(qwen.get("qwen_start_sec"))
        qwen_end = as_float(qwen.get("qwen_end_sec"))
        refined_start = as_float(refined.get("refined_start_sec"))
        refined_end = as_float(refined.get("refined_end_sec"))
        if refined.get("refinement_status") == "skipped" or refined_start is None or refined_end is None:
            skipped += 1
            continue
        processed += 1
        if qwen_start is not None and qwen_end is not None:
            before_durations.append(qwen_end - qwen_start)
            start_shifts.append(abs(refined_start - qwen_start))
            end_shifts.append(abs(refined_end - qwen_end))
            if abs(refined_start - qwen_start) > time_epsilon or abs(refined_end - qwen_end) > time_epsilon:
                changed += 1
        after_durations.append(refined_end - refined_start)
        confidence.append(as_float(refined.get("ranking_confidence")))
        score_margins.append(as_float(refined.get("score_margin")))
        neighbor_separations.append(as_float(refined.get("neighbor_separation")))
        agreement.append(as_float(refined.get("model_agreement_iou")))
        start_uncertainty.append(as_float(refined.get("start_uncertainty")))
        end_uncertainty.append(as_float(refined.get("end_uncertainty")))

    metrics = {
        "number_of_narrations_processed": processed,
        "number_of_narrations_skipped": skipped,
        "percentage_of_segments_changed": float(changed / processed) if processed else 0.0,
        "average_absolute_start_shift": float(np.mean(start_shifts)) if start_shifts else None,
        "average_absolute_end_shift": float(np.mean(end_shifts)) if end_shifts else None,
        "average_duration_before_refinement": float(np.mean(before_durations)) if before_durations else None,
        "average_duration_after_refinement": float(np.mean(after_durations)) if after_durations else None,
        "duration_distribution_before": summary_stats(before_durations),
        "duration_distribution_after": summary_stats(after_durations),
        "confidence_distribution": summary_stats(confidence),
        "score_margin_distribution": summary_stats(score_margins),
        "neighbor_separation_distribution": summary_stats(neighbor_separations),
        "model_agreement_iou_distribution": summary_stats(agreement),
        "start_uncertainty_distribution": summary_stats(start_uncertainty),
        "end_uncertainty_distribution": summary_stats(end_uncertainty),
    }

    if score_records is not None:
        metrics["neighbor_discrimination_accuracy_qwen"] = neighbor_discrimination_accuracy(
            qwen_records,
            score_records,
            "qwen_start_sec",
            "qwen_end_sec",
        )
        metrics["neighbor_discrimination_accuracy_refined"] = neighbor_discrimination_accuracy(
            refined_records,
            score_records,
            "refined_start_sec",
            "refined_end_sec",
        )

    return metrics


def write_metrics(path, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
