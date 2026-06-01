import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from avion.refinement.diagnostics import (
    candidate_neighbor_success,
    find_nearest_candidate,
    recall_by_iou_threshold,
    summary_stats,
    write_metrics,
)
from avion.refinement.candidates import temporal_iou
from avion.refinement.io import as_float, as_int, iter_jsonl, iter_manifest, load_annotation_segments, record_key


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segment ranking refinement diagnostics.")
    parser.add_argument("--input", required=True, help="Refined manifest JSONL")
    parser.add_argument("--qwen-input", required=True, help="Original Qwen-refined manifest")
    parser.add_argument("--scores", default=None, help="Optional scored candidates JSONL")
    parser.add_argument("--test-annotations", default=None, help="Optional test annotation CSV keyed by UUID")
    parser.add_argument("--annotation-output", default=None, help="Optional per-annotation IoU JSONL output")
    parser.add_argument("--iou-thresholds", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated IoU thresholds for r@IoU")
    parser.add_argument("--annotation-uuid-column", default="uuid", help="UUID column in --test-annotations")
    parser.add_argument("--annotation-start-column", default="start_s", help="Start-time column in --test-annotations")
    parser.add_argument("--annotation-end-column", default="end_s", help="End-time column in --test-annotations")
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    return parser.parse_args()


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_annotation_index(args):
    annotations = load_annotation_segments(
        args.test_annotations,
        uuid_column=args.annotation_uuid_column,
        start_column=args.annotation_start_column,
        end_column=args.annotation_end_column,
    )
    by_uuid = defaultdict(list)
    by_uuid_row = {}
    for annotation in annotations:
        by_uuid[str(annotation["uuid"])].append(annotation)
        by_uuid_row[(str(annotation["uuid"]), int(annotation["row_index"]))] = annotation
    return annotations, by_uuid, by_uuid_row


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    start_time = time.time()
    logging.info("evaluate_ranking: starting")
    logging.info(
        "input=%s qwen_input=%s scores=%s test_annotations=%s annotation_output=%s output=%s",
        args.input,
        args.qwen_input,
        args.scores,
        args.test_annotations,
        args.annotation_output,
        args.output,
    )
    logging.info("streaming refined manifest and diagnostics")
    refined_iter = iter_manifest(args.input)
    score_iter = iter_jsonl(args.scores) if args.scores else None
    iou_thresholds = parse_float_list(args.iou_thresholds)
    annotations = []
    annotations_by_uuid = {}
    annotations_by_uuid_row = {}
    if args.test_annotations:
        annotations, annotations_by_uuid, annotations_by_uuid_row = build_annotation_index(args)
        logging.info(
            "loaded test annotations: rows=%d unique_uuids=%d path=%s",
            len(annotations),
            len(annotations_by_uuid),
            args.test_annotations,
        )

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
    qwen_neighbor_successes = []
    refined_neighbor_successes = []
    qwen_annotation_ious = []
    refined_annotation_ious = []
    evaluated_annotation_keys = set()
    evaluated_annotation_uuids = set()

    annotation_file = None
    if args.annotation_output and args.test_annotations:
        Path(args.annotation_output).parent.mkdir(parents=True, exist_ok=True)
        annotation_file = open(args.annotation_output, "w")

    try:
        for idx, refined in enumerate(refined_iter, 1):
            score_record = next(score_iter) if score_iter is not None else None
            refined_key = str(record_key(refined))
            if score_record is not None:
                score_key = str(score_record.get("record_key") or record_key(score_record.get("record") or score_record))
                if score_key != refined_key:
                    logging.warning("score/refined key mismatch at row %d: score=%s refined=%s", idx, score_key, refined_key)

            qwen_start = as_float(refined.get("qwen_start_sec"))
            qwen_end = as_float(refined.get("qwen_end_sec"))
            refined_start = as_float(refined.get("refined_start_sec"))
            refined_end = as_float(refined.get("refined_end_sec"))
            if refined.get("refinement_status") == "skipped" or refined_start is None or refined_end is None:
                skipped += 1
            else:
                processed += 1
                if qwen_start is not None and qwen_end is not None:
                    before_durations.append(qwen_end - qwen_start)
                    start_shifts.append(abs(refined_start - qwen_start))
                    end_shifts.append(abs(refined_end - qwen_end))
                    if abs(refined_start - qwen_start) > 1e-6 or abs(refined_end - qwen_end) > 1e-6:
                        changed += 1
                after_durations.append(refined_end - refined_start)
                confidence.append(as_float(refined.get("ranking_confidence")))
                score_margins.append(as_float(refined.get("score_margin")))
                neighbor_separations.append(as_float(refined.get("neighbor_separation")))
                agreement.append(as_float(refined.get("model_agreement_iou")))
                start_uncertainty.append(as_float(refined.get("start_uncertainty")))
                end_uncertainty.append(as_float(refined.get("end_uncertainty")))

            if score_record is not None:
                candidates = score_record.get("candidates") or []
                qwen_candidate = find_nearest_candidate(candidates, qwen_start, qwen_end)
                refined_candidate = find_nearest_candidate(candidates, refined_start, refined_end)
                qwen_success = candidate_neighbor_success(qwen_candidate) if qwen_candidate is not None else None
                refined_success = candidate_neighbor_success(refined_candidate) if refined_candidate is not None else None
                if qwen_success is not None:
                    qwen_neighbor_successes.append(qwen_success)
                if refined_success is not None:
                    refined_neighbor_successes.append(refined_success)

            if annotations_by_uuid:
                refined_uuid = str(refined.get("test_annotation_uuid") or refined_key)
                refined_row_index = as_int(refined.get("test_annotation_row_index"))
                if refined_row_index is not None and (refined_uuid, refined_row_index) in annotations_by_uuid_row:
                    row_annotations = [annotations_by_uuid_row[(refined_uuid, refined_row_index)]]
                else:
                    row_annotations = annotations_by_uuid.get(refined_uuid, [])
                for annotation in row_annotations:
                    annotation_key = (annotation["uuid"], annotation["row_index"])
                    if annotation_key in evaluated_annotation_keys:
                        continue
                    evaluated_annotation_keys.add(annotation_key)
                    evaluated_annotation_uuids.add(annotation["uuid"])
                    annotation_segment = (annotation["start_sec"], annotation["end_sec"])
                    qwen_iou = None
                    refined_iou = None
                    if qwen_start is not None and qwen_end is not None:
                        qwen_iou = temporal_iou((qwen_start, qwen_end), annotation_segment)
                        qwen_annotation_ious.append(qwen_iou)
                    if refined_start is not None and refined_end is not None:
                        refined_iou = temporal_iou((refined_start, refined_end), annotation_segment)
                        refined_annotation_ious.append(refined_iou)
                    if annotation_file is not None:
                        annotation_file.write(
                            json.dumps(
                                {
                                    "uuid": annotation["uuid"],
                                    "video_uid": annotation.get("video_uid") or refined.get("video_uid"),
                                    "text": refined.get("text"),
                                    "annotation_text": annotation.get("text"),
                                    "annotation_start_sec": annotation["start_sec"],
                                    "annotation_end_sec": annotation["end_sec"],
                                    "qwen_start_sec": qwen_start,
                                    "qwen_end_sec": qwen_end,
                                    "refined_start_sec": refined_start,
                                    "refined_end_sec": refined_end,
                                    "qwen_iou": qwen_iou,
                                    "refined_iou": refined_iou,
                                    "qwen_recall_at_iou": {
                                        f"r@{threshold:g}": bool(qwen_iou is not None and qwen_iou >= threshold)
                                        for threshold in iou_thresholds
                                    },
                                    "refined_recall_at_iou": {
                                        f"r@{threshold:g}": bool(refined_iou is not None and refined_iou >= threshold)
                                        for threshold in iou_thresholds
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            if idx % 100000 == 0:
                logging.info(
                    "evaluation progress: rows=%d processed=%d skipped=%d changed=%d annotation_rows=%d elapsed=%.1fs",
                    idx,
                    processed,
                    skipped,
                    changed,
                    len(evaluated_annotation_keys),
                    time.time() - start_time,
                )
    finally:
        if annotation_file is not None:
            annotation_file.close()

    def accuracy(successes):
        if not successes:
            return {"count": 0, "accuracy": None}
        return {"count": len(successes), "accuracy": float(sum(successes) / len(successes))}

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
    if args.scores:
        metrics["neighbor_discrimination_accuracy_qwen"] = accuracy(qwen_neighbor_successes)
        metrics["neighbor_discrimination_accuracy_refined"] = accuracy(refined_neighbor_successes)
    if args.test_annotations:
        annotation_uuid_set = set(annotations_by_uuid)
        duplicate_uuid_count = sum(1 for rows in annotations_by_uuid.values() if len(rows) > 1)
        metrics["test_annotation_iou_recall"] = {
            "annotation_path": args.test_annotations,
            "annotation_output": args.annotation_output,
            "annotation_rows": len(annotations),
            "unique_annotation_uuids": len(annotation_uuid_set),
            "duplicate_annotation_uuid_count": duplicate_uuid_count,
            "evaluated_annotation_rows": len(evaluated_annotation_keys),
            "evaluated_unique_uuids": len(evaluated_annotation_uuids),
            "missing_annotation_rows": len(annotations) - len(evaluated_annotation_keys),
            "missing_unique_uuids": len(annotation_uuid_set - evaluated_annotation_uuids),
            "iou_thresholds": iou_thresholds,
            "qwen_iou_distribution": summary_stats(qwen_annotation_ious),
            "refined_iou_distribution": summary_stats(refined_annotation_ious),
            "qwen_recall_by_iou": recall_by_iou_threshold(qwen_annotation_ious, iou_thresholds),
            "refined_recall_by_iou": recall_by_iou_threshold(refined_annotation_ious, iou_thresholds),
        }

    write_metrics(args.output, metrics)
    logging.info(
        "evaluate_ranking: done processed=%s skipped=%s changed=%.4f output=%s elapsed=%.1fs",
        metrics.get("number_of_narrations_processed"),
        metrics.get("number_of_narrations_skipped"),
        metrics.get("percentage_of_segments_changed", 0.0),
        args.output,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
