# Segment Ranking Refinement

This pipeline refines Ego4D narration segments by ranking nearby candidate windows with trained video-text alignment checkpoints. It does not train a student model or boundary refiner.

## Manifest Mapping

JSONL/JSON input records are preserved and canonicalized with these field aliases:

| Canonical field | Accepted aliases |
| --- | --- |
| `video_uid` | `video_uid`, `video_id`, `vid` |
| `clip_uid` | `clip_uid`, `clip_id` |
| `narration_uid` | `narration_uid`, `uuid`, `uid`, `id` |
| `text` | `text`, `caption`, `narration`, `clip_text`, `query` |
| `timestamp_sec` | `timestamp_sec`, `timestamp`, `time_sec`, `global_timestamp_sec`; if missing, midpoint of the segment |
| `qwen_start_sec` | `qwen_start_sec`, `qwen_start`, `start_sec`, `start`, `global_start`, `video_start_sec`, `clip_start_sec` |
| `qwen_end_sec` | `qwen_end_sec`, `qwen_end`, `end_sec`, `end`, `global_end`, `video_end_sec`, `clip_end_sec` |

The existing repository Qwen-refined pickle format is also supported:

```text
(uuid, video_id, start, end, caption)
```

It maps to:

```text
narration_uid = uuid
video_uid = video_id
qwen_start_sec = start
qwen_end_sec = end
text = caption
timestamp_sec = (start + end) / 2
source = qwen3vl
```

Plain Ego4D pickle rows `(video_id, start, end, caption)` are mapped the same way, with a synthetic narration id.

## Commands

```bash
python tools/refinement/build_candidates.py \
  --input data/qwen_refined_manifest.jsonl \
  --output outputs/segment_ranking/iter_001/candidates.jsonl \
  --config configs/refinement/segment_ranking.yaml \
  --video-durations data/video_lengths.json
```

```bash
python tools/refinement/score_segments.py \
  --candidates outputs/segment_ranking/iter_001/candidates.jsonl \
  --lavila-checkpoint path/to/lavila_checkpoint.pt \
  --egovlp-checkpoint path/to/egovlp_checkpoint.pt \
  --video-root /path/to/ego4d/video_320px_15sec \
  --output outputs/segment_ranking/iter_001/scored_candidates.jsonl \
  --config configs/refinement/segment_ranking.yaml
```

```bash
python tools/refinement/rank_segments.py \
  --scores outputs/segment_ranking/iter_001/scored_candidates.jsonl \
  --input data/qwen_refined_manifest.jsonl \
  --output outputs/segment_ranking/iter_001/refined_manifest.jsonl \
  --candidate-output outputs/segment_ranking/iter_001/candidate_rankings.jsonl \
  --config configs/refinement/segment_ranking.yaml
```

```bash
python tools/refinement/evaluate_ranking.py \
  --input outputs/segment_ranking/iter_001/refined_manifest.jsonl \
  --qwen-input data/qwen_refined_manifest.jsonl \
  --scores outputs/segment_ranking/iter_001/scored_candidates.jsonl \
  --output outputs/segment_ranking/iter_001/metrics.json
```

If only one checkpoint is supplied, the scorer writes only that model's similarities and ranking falls back to `lavila_only` or `egovlp_only`. With both checkpoints and the default config, ranking uses `lavila_egovlp_ensemble`.

