import csv
import json
import logging
import pickle
from pathlib import Path


LOGGER = logging.getLogger(__name__)


VIDEO_UID_KEYS = ("video_uid", "video_id", "vid")
CLIP_UID_KEYS = ("clip_uid", "clip_id")
NARRATION_UID_KEYS = ("narration_uid", "uuid", "uid", "id")
TEXT_KEYS = ("text", "caption", "narration", "clip_text", "query")
TIMESTAMP_KEYS = ("timestamp_sec", "timestamp", "time_sec", "global_timestamp_sec")
START_KEYS = (
    "qwen_start_sec",
    "qwen_start",
    "start_sec",
    "start",
    "global_start",
    "video_start_sec",
    "clip_start_sec",
)
END_KEYS = (
    "qwen_end_sec",
    "qwen_end",
    "end_sec",
    "end",
    "global_end",
    "video_end_sec",
    "clip_end_sec",
)
DURATION_KEYS = ("video_duration_sec", "duration_sec", "video_duration")


def first_present(record, keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def canonicalize_record(record, index=0):
    if isinstance(record, dict):
        out = dict(record)
        video_uid = first_present(record, VIDEO_UID_KEYS)
        clip_uid = first_present(record, CLIP_UID_KEYS)
        narration_uid = first_present(record, NARRATION_UID_KEYS)
        text = first_present(record, TEXT_KEYS)
        start = as_float(first_present(record, START_KEYS))
        end = as_float(first_present(record, END_KEYS))
        timestamp = as_float(first_present(record, TIMESTAMP_KEYS))
        duration = as_float(first_present(record, DURATION_KEYS))
    elif isinstance(record, (list, tuple)):
        if len(record) >= 5:
            narration_uid, video_uid, start, end, text = record[:5]
            clip_uid = None
        elif len(record) >= 4:
            video_uid, start, end, text = record[:4]
            narration_uid = None
            clip_uid = None
        else:
            raise ValueError(f"Unsupported tuple manifest row at index {index}: {record!r}")
        out = {
            "narration_uid": narration_uid,
            "video_uid": video_uid,
            "clip_uid": clip_uid,
            "qwen_start_sec": start,
            "qwen_end_sec": end,
            "text": text,
            "source": "qwen3vl",
        }
        timestamp = None
        duration = None
        start = as_float(start)
        end = as_float(end)
    else:
        raise TypeError(f"Unsupported manifest row type at index {index}: {type(record)}")

    start = as_float(start)
    end = as_float(end)
    if timestamp is None and start is not None and end is not None:
        timestamp = 0.5 * (start + end)

    if not narration_uid:
        narration_uid = f"{video_uid or 'unknown'}:{index}"

    out["video_uid"] = video_uid
    out["clip_uid"] = clip_uid
    out["narration_uid"] = narration_uid
    out["text"] = text
    out["timestamp_sec"] = timestamp
    out["qwen_start_sec"] = start
    out["qwen_end_sec"] = end
    if duration is not None:
        out["video_duration_sec"] = duration
    out["_record_index"] = index
    return out


def public_record(record):
    return {k: v for k, v in record.items() if not k.startswith("_")}


def read_json_records(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "data", "narrations", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"JSON manifest {path} must be a list or contain a records/data list")


def read_jsonl_records(path):
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def read_manifest(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            raw_records = pickle.load(f)
    elif suffix == ".jsonl":
        raw_records = read_jsonl_records(path)
    elif suffix == ".json":
        raw_records = read_json_records(path)
    else:
        raise ValueError(f"Unsupported manifest suffix for {path}")
    return [canonicalize_record(record, index=i) for i, record in enumerate(raw_records)]


def iter_manifest(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            raw_records = pickle.load(f)
        for idx, record in enumerate(raw_records):
            yield canonicalize_record(record, index=idx)
    elif suffix == ".jsonl":
        for idx, record in enumerate(iter_jsonl(path)):
            yield canonicalize_record(record, index=idx)
    elif suffix == ".json":
        for idx, record in enumerate(read_json_records(path)):
            yield canonicalize_record(record, index=idx)
    else:
        raise ValueError(f"Unsupported manifest suffix for {path}")


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def load_video_durations(path):
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Video durations file must be a JSON object mapping video_uid to seconds")
    return {str(k): as_float(v) for k, v in data.items() if as_float(v) is not None}


def load_uuid_allowlist(path, column="uuid"):
    if not path:
        return None
    allowed = set()
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True
        if has_header:
            reader = csv.DictReader(f)
            if column not in (reader.fieldnames or []):
                raise ValueError(f"UUID allowlist column '{column}' not found in {path}; columns={reader.fieldnames}")
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    allowed.add(value)
        else:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    allowed.add(row[0].strip())
    return allowed


def load_annotation_segments(
    path,
    uuid_column="uuid",
    start_column="start_s",
    end_column="end_s",
    video_column="video_id",
    text_column="caption",
):
    if not path:
        return []
    annotations = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = [uuid_column, start_column, end_column]
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise ValueError(f"Annotation columns {missing} not found in {path}; columns={fieldnames}")
        for row_index, row in enumerate(reader, 1):
            uuid = (row.get(uuid_column) or "").strip()
            start = as_float(row.get(start_column))
            end = as_float(row.get(end_column))
            if not uuid or start is None or end is None:
                continue
            annotations.append(
                {
                    "uuid": uuid,
                    "video_uid": (row.get(video_column) or "").strip() if video_column in fieldnames else None,
                    "start_sec": start,
                    "end_sec": end,
                    "text": row.get(text_column) if text_column in fieldnames else None,
                    "row_index": row_index,
                }
            )
    return annotations


def normalize_match_text(value):
    return " ".join(str(value or "").split())


def segment_iou(first_start, first_end, second_start, second_end):
    first_start = as_float(first_start)
    first_end = as_float(first_end)
    second_start = as_float(second_start)
    second_end = as_float(second_end)
    if None in (first_start, first_end, second_start, second_end):
        return 0.0
    if first_end <= first_start or second_end <= second_start:
        return 0.0
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def segment_gap(first_start, first_end, second_start, second_end):
    first_start = as_float(first_start)
    first_end = as_float(first_end)
    second_start = as_float(second_start)
    second_end = as_float(second_end)
    if None in (first_start, first_end, second_start, second_end):
        return float("inf")
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0.0


def match_records_to_annotations(records, annotations):
    records_by_video = {}
    for record in records:
        records_by_video.setdefault(record.get("video_uid"), []).append(record)

    matched_records = []
    stats = {
        "annotation_rows": len(annotations),
        "matched_rows": 0,
        "unmatched_rows": 0,
        "exact_caption_overlap": 0,
        "exact_caption_no_overlap": 0,
        "video_time_fallback_overlap": 0,
        "video_time_fallback_no_overlap": 0,
    }

    for annotation in annotations:
        video_records = records_by_video.get(annotation.get("video_uid")) or []
        if not video_records:
            stats["unmatched_rows"] += 1
            continue

        annotation_text = normalize_match_text(annotation.get("text"))
        exact_text_records = [
            record
            for record in video_records
            if annotation_text and normalize_match_text(record.get("text")) == annotation_text
        ]
        pool = exact_text_records or video_records
        ann_start = annotation.get("start_sec")
        ann_end = annotation.get("end_sec")
        ann_mid = 0.5 * (ann_start + ann_end)

        def match_key(record):
            start = as_float(record.get("qwen_start_sec"))
            end = as_float(record.get("qwen_end_sec"))
            mid = 0.5 * (start + end) if start is not None and end is not None else float("inf")
            iou = segment_iou(start, end, ann_start, ann_end)
            gap = segment_gap(start, end, ann_start, ann_end)
            return (iou, -gap, -abs(mid - ann_mid))

        best = max(pool, key=match_key)
        best_start = as_float(best.get("qwen_start_sec"))
        best_end = as_float(best.get("qwen_end_sec"))
        best_iou = segment_iou(best_start, best_end, ann_start, ann_end)
        best_gap = segment_gap(best_start, best_end, ann_start, ann_end)
        if exact_text_records:
            strategy = "exact_caption_overlap" if best_iou > 0 else "exact_caption_no_overlap"
        else:
            strategy = "video_time_fallback_overlap" if best_iou > 0 else "video_time_fallback_no_overlap"

        out = dict(best)
        out["source_narration_uid"] = best.get("narration_uid")
        out["source_record_index"] = best.get("_record_index")
        out["narration_uid"] = annotation["uuid"]
        out["test_annotation_uuid"] = annotation["uuid"]
        out["test_annotation_row_index"] = annotation["row_index"]
        out["test_annotation_start_sec"] = ann_start
        out["test_annotation_end_sec"] = ann_end
        out["test_annotation_text"] = annotation.get("text")
        out["test_annotation_match_strategy"] = strategy
        out["test_annotation_match_iou"] = best_iou
        out["test_annotation_match_gap"] = best_gap
        out["_record_index"] = annotation["row_index"]
        matched_records.append(out)
        stats["matched_rows"] += 1
        stats[strategy] += 1

    return matched_records, stats


def attach_video_durations(records, durations):
    if not durations:
        return records
    for record in records:
        video_uid = record.get("video_uid")
        if video_uid in durations:
            record["video_duration_sec"] = durations[video_uid]
    return records


def add_neighbors(records):
    groups = {}
    for record in records:
        group_key = record.get("clip_uid") or record.get("video_uid") or "__missing_video__"
        groups.setdefault(group_key, []).append(record)

    for group in groups.values():
        group.sort(
            key=lambda r: (
                as_float(r.get("timestamp_sec"), float("inf")),
                as_float(r.get("qwen_start_sec"), float("inf")),
                as_int(r.get("_record_index"), 0),
            )
        )
        for idx, record in enumerate(group):
            prev_record = group[idx - 1] if idx > 0 else None
            next_record = group[idx + 1] if idx + 1 < len(group) else None
            if prev_record is not None:
                record["previous_narration_uid"] = prev_record.get("narration_uid")
                record["previous_timestamp_sec"] = prev_record.get("timestamp_sec")
                record["previous_text"] = prev_record.get("text")
            if next_record is not None:
                record["next_narration_uid"] = next_record.get("narration_uid")
                record["next_timestamp_sec"] = next_record.get("timestamp_sec")
                record["next_text"] = next_record.get("text")
    records.sort(key=lambda r: as_int(r.get("_record_index"), 0))
    return records


def record_key(record):
    return str(record.get("narration_uid") or record.get("_record_index"))
