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
from avion.refinement.io import iter_jsonl, public_record, record_key
from avion.refinement.ranking import fallback_result, rank_scored_record


def parse_args():
    parser = argparse.ArgumentParser(description="Rank scored candidate segments and export refined manifest.")
    parser.add_argument("--scores", required=True, help="Scored candidates JSONL")
    parser.add_argument("--input", default=None, help="Original Qwen-refined manifest; kept for CLI compatibility")
    parser.add_argument("--output", required=True, help="Output refined manifest JSONL")
    parser.add_argument("--candidate-output", default=None, help="Optional top candidate rankings JSONL")
    parser.add_argument("--config", default=None, help="Refinement YAML config")
    parser.add_argument("--log-every", type=int, default=10000, help="Log progress every N records")
    parser.add_argument("--start-index", type=int, default=0, help="Optional 0-based inclusive scored-record start index")
    parser.add_argument("--end-index", type=int, default=None, help="Optional 0-based exclusive scored-record end index")
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
    logging.info("rank_segments: starting")
    logging.info("scores=%s input=%s output=%s config=%s", args.scores, args.input, args.output, args.config)
    logging.info("range start_index=%s end_index=%s", args.start_index, args.end_index)
    config = load_config(args.config)

    logging.info("streaming scored candidates from %s", args.scores)
    skipped = 0
    changed = 0
    num_ranked = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ranking_file = None
    if args.candidate_output and config["output"].get("store_candidate_rankings", True):
        Path(args.candidate_output).parent.mkdir(parents=True, exist_ok=True)
        ranking_file = open(args.candidate_output, "w")

    logging.info("writing refined manifest to %s", args.output)
    try:
        with open(args.output, "w") as refined_file:
            for zero_idx, score_record in enumerate(iter_jsonl(args.scores)):
                if zero_idx < args.start_index:
                    continue
                if args.end_index is not None and zero_idx >= args.end_index:
                    break

                num_ranked += 1
                original = score_record.get("record") or score_record
                key = str(score_record.get("record_key") or record_key(original))
                result, ranking_rows = rank_scored_record(score_record, config)
                if result.get("refinement_status") == "skipped":
                    skipped += 1

                out_record = public_record(original)
                out_record.update(result)
                if (
                    out_record.get("refined_start_sec") != out_record.get("qwen_start_sec")
                    or out_record.get("refined_end_sec") != out_record.get("qwen_end_sec")
                ):
                    changed += 1
                refined_file.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                if ranking_file is not None:
                    ranking_file.write(
                        json.dumps(
                            {
                                "record_key": key,
                                "video_uid": out_record.get("video_uid"),
                                "clip_uid": out_record.get("clip_uid"),
                                "narration_uid": out_record.get("narration_uid"),
                                "text": out_record.get("text"),
                                "previous_text": score_record.get("previous_text") if score_record else None,
                                "next_text": score_record.get("next_text") if score_record else None,
                                "timestamp_sec": out_record.get("timestamp_sec"),
                                "qwen_start_sec": out_record.get("qwen_start_sec"),
                                "qwen_end_sec": out_record.get("qwen_end_sec"),
                                "refined_start_sec": out_record.get("refined_start_sec"),
                                "refined_end_sec": out_record.get("refined_end_sec"),
                                "lavila_best_start_sec": out_record.get("lavila_best_start_sec"),
                                "lavila_best_end_sec": out_record.get("lavila_best_end_sec"),
                                "egovlp_best_start_sec": out_record.get("egovlp_best_start_sec"),
                                "egovlp_best_end_sec": out_record.get("egovlp_best_end_sec"),
                                "ranking_confidence": out_record.get("ranking_confidence"),
                                "update_reason": out_record.get("update_reason"),
                                "top_candidates": ranking_rows,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if args.log_every > 0 and num_ranked % args.log_every == 0:
                    logging.info(
                        "rank progress: %d/%s records, changed=%d skipped=%d elapsed=%.1fs",
                        num_ranked,
                        "unknown",
                        changed,
                        skipped,
                        time.time() - start_time,
                    )
    finally:
        if ranking_file is not None:
            ranking_file.close()

    if ranking_file is not None:
        logging.info("wrote candidate rankings to %s", args.candidate_output)
    logging.info(
        "rank_segments: done ranked=%d changed=%d skipped=%d output=%s elapsed=%.1fs",
        num_ranked,
        changed,
        skipped,
        args.output,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
