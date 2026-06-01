import argparse
import json
import logging
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avion.refinement.candidates import build_candidate_record
from avion.refinement.config import load_config
from avion.refinement.io import (
    add_neighbors,
    attach_video_durations,
    load_annotation_segments,
    load_uuid_allowlist,
    load_video_durations,
    match_records_to_annotations,
    read_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build temporal refinement candidates.")
    parser.add_argument("--input", required=True, help="Qwen-refined narration manifest (.jsonl/.json/.pkl)")
    parser.add_argument("--output", required=True, help="Output candidates JSONL")
    parser.add_argument("--config", default=None, help="Refinement YAML config")
    parser.add_argument("--video-durations", default=None, help="Optional JSON map video_uid -> duration seconds")
    parser.add_argument("--uuid-filter-csv", default=None, help="Optional CSV/list of narration UUIDs to keep")
    parser.add_argument("--uuid-column", default="uuid", help="Column name for --uuid-filter-csv")
    parser.add_argument("--log-every", type=int, default=10000, help="Log progress every N records")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    start_time = time.time()
    logging.info("build_candidates: starting")
    logging.info(
        "input=%s output=%s config=%s uuid_filter_csv=%s uuid_column=%s",
        args.input,
        args.output,
        args.config,
        args.uuid_filter_csv,
        args.uuid_column,
    )
    overrides = {}
    if args.video_durations:
        overrides.setdefault("paths", {})["video_durations"] = args.video_durations
    config = load_config(args.config, overrides)

    logging.info("loading manifest")
    records = read_manifest(args.input)
    logging.info("loaded %d records", len(records))
    logging.info("loading video durations from %s", config["paths"].get("video_durations"))
    durations = load_video_durations(config["paths"].get("video_durations"))
    logging.info("loaded %d video durations", len(durations))
    records = attach_video_durations(records, durations)
    logging.info("attaching previous/next narration context")
    records = add_neighbors(records)
    allowed_uuids = load_uuid_allowlist(args.uuid_filter_csv, args.uuid_column)
    if allowed_uuids is not None:
        before = len(records)
        direct_records = [record for record in records if str(record.get("narration_uid")) in allowed_uuids]
        if direct_records:
            records = direct_records
            logging.info(
                "filtered records by UUID allowlist using narration_uid: before=%d after=%d allowlist=%d path=%s column=%s",
                before,
                len(records),
                len(allowed_uuids),
                args.uuid_filter_csv,
                args.uuid_column,
            )
        else:
            annotations = load_annotation_segments(args.uuid_filter_csv, uuid_column=args.uuid_column)
            records, match_stats = match_records_to_annotations(records, annotations)
            logging.info(
                "filtered records by annotation-field matching because no narration_uid matched UUID allowlist: "
                "before=%d after=%d allowlist=%d path=%s stats=%s",
                before,
                len(records),
                len(allowed_uuids),
                args.uuid_filter_csv,
                match_stats,
            )

    num_candidates = 0
    skipped = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    logging.info("building and streaming candidates to %s", args.output)
    with open(args.output, "w") as f:
        for idx, record in enumerate(records, 1):
            candidate_record = build_candidate_record(record, config)
            num_candidates += len(candidate_record.get("candidates") or [])
            if candidate_record.get("errors"):
                skipped += 1
            f.write(json.dumps(candidate_record, ensure_ascii=False) + "\n")
            if args.log_every > 0 and idx % args.log_every == 0:
                elapsed = time.time() - start_time
                logging.info(
                    "candidate progress: %d/%d records, candidates=%d, warned_or_skipped=%d, elapsed=%.1fs",
                    idx,
                    len(records),
                    num_candidates,
                    skipped,
                    elapsed,
                )

    logging.info(
        "build_candidates: done processed=%d skipped_or_warned=%d candidates=%d output=%s elapsed=%.1fs",
        len(records),
        skipped,
        num_candidates,
        args.output,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
