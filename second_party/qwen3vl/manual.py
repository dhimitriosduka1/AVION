import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK_LEN_SEC_DEFAULT = 15.0


@dataclass(frozen=True)
class Segment:
    video_id: str
    start: float
    end: float
    caption: str


def as_segment(row: Any) -> Segment:
    if isinstance(row, (tuple, list)) and len(row) == 4:
        video_id, start, end, caption = row
        return Segment(str(video_id), float(start), float(end), str(caption))
    if isinstance(row, dict):
        return Segment(
            str(row["video_id"]),
            float(row["start"]),
            float(row["end"]),
            str(row["caption"]),
        )
    raise TypeError(f"Unsupported row: {type(row)}")


def chunk_id_from_time(t: float, chunk_len_sec: float) -> int:
    if t < 0:
        return 0
    return int(math.floor(t / chunk_len_sec) * chunk_len_sec)


def chunk_path(video_root: str, video_id: str, chunk_id: int) -> str:
    return os.path.join(video_root, f"{video_id}.mp4", f"{chunk_id}.mp4")


class Ego4DChunkedTemporalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pkl_path: str,
        video_root: str,
        fps: int,
        chunk_len_sec: float = CHUNK_LEN_SEC_DEFAULT,
        strict_exists: bool = True,
        only_video_id: Optional[str] = None,
    ) -> None:
        self.video_root = video_root
        self.fps = int(fps)
        self.chunk_len_sec = float(chunk_len_sec)
        self.strict_exists = bool(strict_exists)

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
            rows = raw["data"]
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = list(raw)

        segments = [as_segment(r) for r in rows]
        grouped: Dict[str, List[Segment]] = {}
        for s in segments:
            if only_video_id is not None and s.video_id != only_video_id:
                continue
            grouped.setdefault(s.video_id, []).append(s)

        for vid in grouped:
            grouped[vid].sort(key=lambda x: x.start)

        self._groups = grouped
        self._index: List[Tuple[str, int]] = []
        for vid in sorted(grouped.keys()):
            for i in range(len(grouped[vid])):
                self._index.append((vid, i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id, within_idx = self._index[idx]
        seg = self._groups[video_id][within_idx]

        seed_center = 0.5 * (seg.start + seg.end)
        chunk_id = chunk_id_from_time(seed_center, self.chunk_len_sec)
        vp = chunk_path(self.video_root, video_id, chunk_id)

        if self.strict_exists and not os.path.exists(vp):
            raise FileNotFoundError(vp)

        seed_start_rel = seg.start - float(chunk_id)
        seed_end_rel = seg.end - float(chunk_id)

        return {
            "video_id": video_id,
            "caption": seg.caption,
            "global_start": seg.start,
            "global_end": seg.end,
            "chunk_id": chunk_id,
            "video_path": vp,
            "seed_start": seed_start_rel,
            "seed_end": seed_end_rel,
            "fps": self.fps,
        }


def build_prompt(*, caption: str, seed_start: float, seed_end: float) -> str:
    text_template = """\
You are given an egocentric (first-person) video. Your task is temporal localization for the captioned action.

Caption:
"{caption}"

Seed timestamp, the center of the segment: {seed_center} seconds.

What to do:
- Briefly describe the visible scene (very short).
- Find the tightest time interval that FULLY contains the action described by the caption.
- Use only visible evidence: the camera wearer's hands, manipulated objects, contact events, and object motion caused by the wearer.
- Do NOT assume the seed is correct; it is only a hint where to look.

Boundary definitions:
- start = first moment the action clearly begins (reach/grasp begins OR object starts moving due to the wearer).
- end   = first moment the action is completed and the outcome is stable (release/placement completed; object comes to rest; hands disengage).

STRICT OUTPUT RULES:
- Return ONLY valid JSON, then a newline, then the literal token END.
- No markdown, no comments, no trailing commas.
- Keep it short:
  - scene_summary: <= 20 words
  - notes: <= 20 words (or empty string)
  - evidence: list of <= 3 short object strings (e.g., ["bowl", "counter", "hand"])
- start/end must be numbers in seconds (floats allowed).

JSON schema (exact keys):
scene_summary: string
caption: string
start: number
end: number
confidence: number (0.0 to 1.0)
evidence: list of strings
notes: string

JSON:
"""
    return text_template.format(
        caption=caption,
        seed_center=0.5 * (seed_start + seed_end),
    )


def build_messages(*, video_path: str, fps: int, prompt_text: str):
    video_item = {"type": "video", "video": video_path, "fps": float(fps)}
    return [
        {"role": "user", "content": [video_item, {"type": "text", "text": prompt_text}]}
    ]


def unpack_videos(videos):
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        return list(videos), list(video_metadatas)
    return None, None


def parse_json_until_end(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.endswith("END"):
        s = s[: -len("END")].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in output: {text[:400]}")
    return json.loads(s[start : end + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pkl_path",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl",
    )
    ap.add_argument(
        "--video_root",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/ff6cfcfc-0441-4963-9bab-9eca357470fa.mp4",
    )
    ap.add_argument("--model_path", type=str, default=MODEL_PATH_DEFAULT)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunk_len_sec", type=float, default=CHUNK_LEN_SEC_DEFAULT)
    ap.add_argument(
        "--only_video_id", type=str, default="ff6cfcfc-0441-4963-9bab-9eca357470fa"
    )
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=2056)
    ap.add_argument("--strict_exists", action="store_true")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )

    dataset = Ego4DChunkedTemporalDataset(
        pkl_path=args.pkl_path,
        video_root=args.video_root,
        fps=args.fps,
        chunk_len_sec=args.chunk_len_sec,
        strict_exists=bool(args.strict_exists),
        only_video_id=args.only_video_id,
    )

    n = (
        len(dataset)
        if args.max_samples is None or args.max_samples < 0
        else min(len(dataset), args.max_samples)
    )

    results = []
    for i in range(n):
        item = dataset[i]
        prompt_text = build_prompt(
            caption=item["caption"],
            seed_start=item["seed_start"],
            seed_end=item["seed_end"],
        )
        messages = build_messages(
            video_path=item["video_path"],
            fps=item["fps"],
            prompt_text=prompt_text,
        )

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        videos, video_metadatas = unpack_videos(videos)

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(model.device)

        generated_ids = model.generate(
            **inputs, max_new_tokens=int(args.max_new_tokens)
        )
        gen_only = generated_ids[:, inputs["input_ids"].shape[1] :]
        out = processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        pred = parse_json_until_end(out)

        pred_start = float(pred["start"])
        pred_end = float(pred["end"])
        chunk_id = float(item["chunk_id"])

        results.append(
            {
                "video_id": item["video_id"],
                "caption": item["caption"],
                "chunk_id": item["chunk_id"],
                "chunk_path": item["video_path"],
                "seed_start_chunk": item["seed_start"],
                "seed_end_chunk": item["seed_end"],
                "pred_start_chunk": pred_start,
                "pred_end_chunk": pred_end,
                "pred_start_global": pred_start + chunk_id,
                "pred_end_global": pred_end + chunk_id,
                "confidence": float(pred.get("confidence", 0.0)),
                "scene_summary": pred.get("scene_summary", ""),
                "evidence": pred.get("evidence", []),
                "notes": pred.get("notes", ""),
            }
        )

        # print(json.dumps(result, ensure_ascii=False))
    with open("/u/dduka/project/AVION/o_.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
