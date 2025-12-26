import argparse
import json
import math
import os
import pickle
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from collections import OrderedDict


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
    """Returns the start time of the chunk containing t."""
    if t < 0:
        return 0
    return int(math.floor(t / chunk_len_sec) * chunk_len_sec)


def get_covering_chunk_ids(start: float, end: float, chunk_len_sec: float) -> List[int]:
    """Returns a sorted list of chunk IDs that cover the interval [start, end]."""
    first_chunk = chunk_id_from_time(start, chunk_len_sec)
    last_chunk = chunk_id_from_time(end, chunk_len_sec)

    chunks = []
    curr = first_chunk
    while curr <= last_chunk:
        chunks.append(curr)
        curr += int(chunk_len_sec)
    return chunks


def chunk_path(video_root: str, video_id: str, chunk_id: int) -> str:
    return os.path.join(video_root, f"{video_id}.mp4", f"{chunk_id}.mp4")


def concat_videos_lossless(video_paths: List[str], output_path: str) -> None:
    """
    Concatenates videos using ffmpeg concat demuxer (stream copy).
    This is extremely efficient as it does not re-encode.
    """
    if not video_paths:
        raise ValueError("No video paths provided for concatenation.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file_path = f.name
        for vp in video_paths:
            safe_vp = vp.replace("'", "'\\''")
            f.write(f"file '{safe_vp}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-c",
            "copy",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            output_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(list_file_path):
            os.remove(list_file_path)


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

        chunk_ids = get_covering_chunk_ids(seg.start, seg.end, self.chunk_len_sec)

        valid_chunk_ids = []
        valid_paths = []
        for cid in chunk_ids:
            vp = chunk_path(self.video_root, video_id, cid)
            if os.path.exists(vp):
                valid_chunk_ids.append(cid)
                valid_paths.append(vp)
            elif self.strict_exists:
                raise FileNotFoundError(f"Chunk missing: {vp}")

        if not valid_paths:
            raise FileNotFoundError(
                f"No valid chunks found for {video_id} interval {seg.start}-{seg.end}"
            )

        base_offset = float(valid_chunk_ids[0])
        seed_start_rel = seg.start - base_offset
        seed_end_rel = seg.end - base_offset

        return {
            "video_id": video_id,
            "caption": seg.caption,
            "global_start": seg.start,
            "global_end": seg.end,
            "chunk_ids": valid_chunk_ids,
            "chunk_paths": valid_paths,
            "base_offset": base_offset,
            "seed_start": seed_start_rel,
            "seed_end": seed_end_rel,
            "fps": self.fps,
        }


def build_prompt(*, caption: str, seed_start: float, seed_end: float) -> str:
    text_template = """\
TASK: Temporal localization in egocentric video.

ACTION TO LOCATE: "{caption}"

SEED WINDOW (approximate): {seed_start:.2f}s to {seed_end:.2f}s (use as starting point only, may be inaccurate).

ANALYSIS STEPS:
1. Watch the video and identify the camera wearer's hands throughout.
2. Find when the described action STARTS: the exact moment of first intentional movement toward the action (hand reaches, begins grasp, or object starts moving due to wearer).
3. Find when the action ENDS: the exact moment the action goal is achieved (object released/placed, hands withdraw, result is stable).

VISUAL CUES TO TRACK:
- Hand position and motion relative to target objects
- Object state changes (picked up, moved, opened, closed, placed)
- Contact events (hand touches object, object touches surface)
- Motion ends (object comes to rest, hand stops moving)

CRITICAL RULES:
- Boundaries must be TIGHT: start at first evidence of action, end when action completes
- Do NOT include preparation or aftermath unless part of the described action
- If action spans multiple sub-actions, include the full sequence
- Times are relative to video start (0.0s = first frame)

OUTPUT (JSON only, no explanation):
{{"scene_summary": "<20 words describing setting>", "caption": "{caption}", "start": <seconds>, "end": <seconds>, "confidence": <0.0-1.0>, "evidence": ["<object1>", "<object2>"], "notes": "<brief uncertainty if any>"}}
END"""
    return text_template.format(
        caption=caption, seed_start=seed_start, seed_end=seed_end
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


def init_distributed() -> Tuple[bool, int, int, int]:
    """
    Returns: (is_distributed, rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def gather_results(
    all_local: List[Dict[str, Any]], distributed: bool, rank: int, world_size: int
):
    if not distributed:
        return all_local
    gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, all_local)
    if rank == 0:
        merged: List[Dict[str, Any]] = []
        for part in gathered:
            merged.extend(part)
        return merged
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pkl_path",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_42.pkl",
    )
    ap.add_argument(
        "--video_root",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/",
    )
    ap.add_argument("--model_path", type=str, default=MODEL_PATH_DEFAULT)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunk_len_sec", type=float, default=CHUNK_LEN_SEC_DEFAULT)
    ap.add_argument(
        "--only_video_id", type=str, default="ff6cfcfc-0441-4963-9bab-9eca357470fa"
    )
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=2056)
    ap.add_argument("--output_path", type=str, default="results.json")
    ap.add_argument("--strict_exists", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load model/processor per-rank onto its GPU (no device_map sharding for DDP-style multi-proc)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    dataset = Ego4DChunkedTemporalDataset(
        pkl_path=args.pkl_path,
        video_root=args.video_root,
        fps=args.fps,
        chunk_len_sec=args.chunk_len_sec,
        strict_exists=bool(args.strict_exists),
        only_video_id=args.only_video_id,
    )

    # Optionally cap total samples globally (not per-rank)
    if args.max_samples is not None and args.max_samples >= 0:
        # Truncate deterministically by wrapping with Subset indices [0:max_samples]
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(min(len(dataset), int(args.max_samples)))))

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )

    # batch_size=1 because each item is a video inference
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    results_local: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for j, item in enumerate(tqdm(loader)):
            try:
                chunk_paths = item["chunk_paths"]
                if len(chunk_paths) == 1:
                    current_video_path = chunk_paths[0]
                else:
                    merged_filename = f"merged_rank{rank}_{item['video_id']}_{j}.mp4"
                    merged_path = os.path.join(temp_dir, merged_filename)
                    concat_videos_lossless(chunk_paths, merged_path)
                    current_video_path = merged_path

                prompt_text = build_prompt(
                    caption=item["caption"],
                    seed_start=item["seed_start"],
                    seed_end=item["seed_end"],
                )

                messages = build_messages(
                    video_path=current_video_path,
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
                ).to(device)

                with torch.inference_mode():
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
                base_offset = float(item["base_offset"])

                results_local.append(
                    {
                        "video_id": item["video_id"],
                        "caption": item["caption"],
                        "chunk_ids": item["chunk_ids"],
                        "base_offset": base_offset,
                        "seed_start_rel": item["seed_start"],
                        "seed_end_rel": item["seed_end"],
                        "pred_start_rel": pred_start,
                        "pred_end_rel": pred_end,
                        "pred_start_global": pred_start + base_offset,
                        "pred_end_global": pred_end + base_offset,
                        "confidence": float(pred.get("confidence", 0.0)),
                        "scene_summary": pred.get("scene_summary", ""),
                        "evidence": pred.get("evidence", []),
                        "notes": pred.get("notes", ""),
                    }
                )

                if rank == 0 and (j % 10 == 0):
                    print(
                        f"[rank0] processed {j} samples on rank0 (world_size={world_size})"
                    )

            except Exception as e:
                # Keep going; store an error record (optional)
                results_local.append(
                    {
                        "video_id": item.get("video_id", ""),
                        "caption": item.get("caption", ""),
                        "error": str(e),
                    }
                )

    merged = gather_results(results_local, distributed, rank, world_size)

    if rank == 0:
        # Optional: stable order for final file
        merged.sort(
            key=lambda r: (
                r.get("video_id", ""),
                float(r.get("pred_start_global", 1e18)),
            )
        )
        with open(args.output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Wrote {len(merged)} results to {args.output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
