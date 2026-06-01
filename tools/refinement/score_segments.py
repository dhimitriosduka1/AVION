import argparse
import json
import logging
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avion.refinement.config import load_config
from avion.refinement.io import iter_jsonl
from avion.refinement.model_adapters import EgoVLPScorer, LaViLaScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Score candidate segments with LaViLa and/or EgoVLP.")
    parser.add_argument("--candidates", required=True, help="Candidates JSONL from build_candidates.py")
    parser.add_argument("--lavila-checkpoint", default=None, help="LaViLa checkpoint path")
    parser.add_argument("--egovlp-checkpoint", default=None, help="EgoVLP checkpoint path")
    parser.add_argument("--output", required=True, help="Scored candidates JSONL")
    parser.add_argument("--config", default=None, help="Refinement YAML config")
    parser.add_argument("--video-root", default=None, help="Ego4D chunk video root")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Scoring batch size")
    parser.add_argument("--inference-precision", default=None, help="auto/amp/fp16/bf16/fp32/off")
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N scored records")
    return parser.parse_args()


def build_overrides(args):
    overrides = {}
    if args.video_root:
        overrides.setdefault("paths", {})["video_root"] = args.video_root
    if args.device:
        overrides.setdefault("scoring", {})["device"] = args.device
    if args.batch_size is not None:
        overrides.setdefault("scoring", {})["batch_size"] = args.batch_size
    if args.inference_precision:
        overrides.setdefault("scoring", {})["inference_precision"] = args.inference_precision
    return overrides


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    start_time = time.time()
    logging.info("score_segments: starting")
    logging.info("candidates=%s output=%s config=%s", args.candidates, args.output, args.config)
    logging.info(
        "video_root=%s device=%s batch_size=%s inference_precision=%s",
        args.video_root,
        args.device,
        args.batch_size,
        args.inference_precision,
    )
    config = load_config(args.config, build_overrides(args))

    scorers = []
    if args.lavila_checkpoint:
        try:
            logging.info("loading LaViLa scorer from %s", args.lavila_checkpoint)
            scorers.append(LaViLaScorer(args.lavila_checkpoint, config))
            logging.info("loaded LaViLa checkpoint: %s", args.lavila_checkpoint)
        except Exception as exc:
            logging.exception("LaViLa scoring disabled: %s", exc)
    if args.egovlp_checkpoint:
        try:
            logging.info("loading EgoVLP scorer from %s", args.egovlp_checkpoint)
            scorers.append(EgoVLPScorer(args.egovlp_checkpoint, config))
            logging.info("loaded EgoVLP checkpoint: %s", args.egovlp_checkpoint)
        except Exception as exc:
            logging.exception("EgoVLP scoring disabled: %s", exc)

    if not scorers:
        raise RuntimeError("No scoring models were loaded. Provide a valid LaViLa and/or EgoVLP checkpoint.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_candidates = 0
    total_errors = 0
    num_records = 0
    logging.info("streaming scored candidates to %s", args.output)
    with open(args.output, "w") as f:
        for idx, record in enumerate(iter_jsonl(args.candidates), 1):
            record_start = time.time()
            num_records = idx
            num_candidates = len(record.get("candidates") or [])
            total_candidates += num_candidates
            for scorer in scorers:
                try:
                    scorer.score_record(record)
                except Exception as exc:
                    record.setdefault("errors", []).append(f"{scorer.model_name}_scoring_failure: {exc}")
                    logging.exception("Scoring failure for narration_uid=%s model=%s", record.get("narration_uid"), scorer.model_name)
            if record.get("errors"):
                total_errors += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.log_every > 0 and idx % args.log_every == 0:
                logging.info(
                    "score progress: records=%d candidates=%d records_with_errors=%d last_record=%.2fs elapsed=%.1fs",
                    idx,
                    total_candidates,
                    total_errors,
                    time.time() - record_start,
                    time.time() - start_time,
                )

    logging.info(
        "score_segments: done scored_records=%d candidates=%d records_with_errors=%d output=%s elapsed=%.1fs",
        num_records,
        total_candidates,
        total_errors,
        args.output,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
