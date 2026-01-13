import argparse
import asyncio
import json
import math
import os
import pickle
import subprocess
import tempfile
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from prompts import v1, v2

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK_LEN_SEC_DEFAULT = 15.0
DEFAULT_API_URL = "http://daisg107:30000/v1"

# -----------------------------------------------------------------------------
# Helper Classes & Functions
# -----------------------------------------------------------------------------

prompt_mapper = {
    "v1": v1.template,
    "v2": v2.template,
}


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


def get_covering_chunk_ids(start: float, end: float, chunk_len_sec: float) -> List[int]:
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
    """Concatenates videos using ffmpeg concat demuxer (stream copy)."""
    if not video_paths:
        raise ValueError("No video paths provided for concatenation.")

    # Create list file for ffmpeg
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


def parse_json_until_end(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    if s.endswith("END"):
        s = s[: -len("END")].strip()

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in output: {text[:400]}")
    return json.loads(s[start : end + 1])


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class Ego4DChunkedTemporalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pkl_path: str,
        video_root: str,
        merge_root: str,
        fps: int,
        chunk_len_sec: float = CHUNK_LEN_SEC_DEFAULT,
        strict_exists: bool = True,
        only_video_id: Optional[str] = None,
    ) -> None:
        self.video_root = video_root
        self.merge_root = merge_root
        self.fps = int(fps)
        self.chunk_len_sec = float(chunk_len_sec)
        self.strict_exists = bool(strict_exists)

        print(f"Loading metadata from {pkl_path}...")
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

        # --- MODIFICATION START: Filter Logic with Validation ---
        for s in segments:
            if only_video_id is not None and s.video_id != only_video_id:
                continue
            grouped.setdefault(s.video_id, []).append(s)

        if only_video_id is not None and len(grouped) == 0:
            raise ValueError(
                f"DEBUG ERROR: Video ID '{only_video_id}' was requested, "
                f"but it was not found in the loaded metadata ({len(segments)} total segments)."
            )
        # --- MODIFICATION END ---

        for vid in grouped:
            grouped[vid].sort(key=lambda x: x.start)

        self._groups = grouped
        self._index: List[Tuple[str, int]] = []
        for vid in sorted(grouped.keys()):
            for i in range(len(grouped[vid])):
                self._index.append((vid, i))

        # Ensure merge directory exists
        os.makedirs(self.merge_root, exist_ok=True)

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

        # Logic: Determine the single video path to use
        if len(valid_paths) == 1:
            final_video_path = valid_paths[0]
        else:
            # Merge logic: Create a unique name based on video ID and start chunk
            safe_vid = video_id.replace("/", "_")
            merged_filename = f"merged_{safe_vid}_{valid_chunk_ids[0]}.mp4"
            merged_path = os.path.join(self.merge_root, merged_filename)

            # Only merge if it doesn't already exist to save time
            if not os.path.exists(merged_path):
                concat_videos_lossless(valid_paths, merged_path)

            final_video_path = merged_path

        base_offset = float(valid_chunk_ids[0])
        seed_start_rel = seg.start - base_offset
        seed_end_rel = seg.end - base_offset

        return {
            "video_id": video_id,
            "caption": seg.caption,
            "global_start": seg.start,
            "global_end": seg.end,
            "chunk_ids": valid_chunk_ids,
            "video_path": f"file:{final_video_path}",
            "base_offset": base_offset,
            "seed_start": seed_start_rel,
            "seed_end": seed_end_rel,
            "fps": self.fps,
        }


# -----------------------------------------------------------------------------
# Prompt Logic
# -----------------------------------------------------------------------------


def build_prompt(
    *, caption: str, seed_start: float, seed_end: float, version: str = "v2"
) -> str:
    return prompt_mapper[version].format(
        caption=caption, seed_start=seed_start, seed_end=seed_end
    )


# -----------------------------------------------------------------------------
# Async API Logic
# -----------------------------------------------------------------------------


async def process_sample(client: AsyncOpenAI, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single sample. Note: Semaphore is now handled by the worker logic.
    """
    start_time = time.perf_counter()
    try:
        video_path_to_send = sample["video_path"]
        prompt_text = build_prompt(
            caption=sample["caption"],
            seed_start=sample["seed_start"],
            seed_end=sample["seed_end"],
        )

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "video_url",
                            "video_url": {"url": video_path_to_send},
                        },
                    ],
                }
            ],
            max_tokens=256,
            temperature=0.7,
        )

        raw_content = response.choices[0].message.content
        latency = time.perf_counter() - start_time

        pred = parse_json_until_end(raw_content)

        pred_start = float(pred["start"])
        pred_end = float(pred["end"])
        base_offset = float(sample["base_offset"])

        return {
            "status": "success",
            "video_id": sample["video_id"],
            "caption": sample["caption"],
            "chunk_ids": sample["chunk_ids"],
            "base_offset": base_offset,
            "seed_start_rel": sample["seed_start"],
            "seed_end_rel": sample["seed_end"],
            "pred_start_rel": pred_start,
            "pred_end_rel": pred_end,
            "pred_start_global": pred_start + base_offset,
            "pred_end_global": pred_end + base_offset,
            "confidence": float(pred.get("confidence", 0.0)),
            "scene_summary": pred.get("scene_summary", ""),
            "evidence": pred.get("evidence", []),
            "notes": pred.get("notes", ""),
            "raw_response": raw_content,
            "latency_sec": latency,
        }

    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "status": "error",
            "video_id": sample.get("video_id", "unknown"),
            "caption": sample.get("caption", "unknown"),
            "error": str(e),
            "latency_sec": latency,
            "raw_response": locals().get("raw_content", ""),
        }


# -----------------------------------------------------------------------------
# Producer / Consumer Logic
# -----------------------------------------------------------------------------


def dataset_producer(loader, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Runs in a separate thread. Iterates over the DataLoader (blocking)
    and schedules putting items into the async queue.
    """
    try:
        for sample in loader:
            # This is thread-safe scheduling into the event loop
            future = asyncio.run_coroutine_threadsafe(queue.put(sample), loop)
            # Wait for the item to be accepted by the queue (respects maxsize)
            future.result()
    finally:
        # Signal end of stream
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()


async def worker(
    client: AsyncOpenAI, queue: asyncio.Queue, results_list: List[Dict], pbar: tqdm
):
    """
    Continuously pulls from the queue and processes samples.
    """
    while True:
        sample = await queue.get()
        if sample is None:
            # End of stream signal
            queue.task_done()
            # Propagate None for other workers to exit
            await queue.put(None)
            break

        try:
            res = await process_sample(client, sample)
            results_list.append(res)
        finally:
            pbar.update(1)
            queue.task_done()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_42.pkl",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/",
    )
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--output_path", type=str, default="results_sglang.json")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--concurrency", type=int, default=64, help="Max concurrent API requests"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Workers for ffmpeg merging"
    )
    # --- MODIFIED ARGUMENT HELP ---
    parser.add_argument(
        "--only_video_id",
        type=str,
        default=None,
        help="Debug: Only process this specific video ID. Raises error if ID not found.",
    )
    # ------------------------------
    parser.add_argument("--strict_exists", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    # 1. Setup Client
    print(f"Connecting to SGLang at {args.api_url}")
    client = AsyncOpenAI(base_url=args.api_url, api_key="EMPTY")

    # --- ADDED: Debug logging ---
    if args.only_video_id:
        print(f"\n!!! DEBUG MODE ACTIVE !!!")
        print(f"Filtering dataset for single Video ID: {args.only_video_id}\n")

    with tempfile.TemporaryDirectory() as temp_dir_path:
        print(f"Using temp dir for merged videos: {temp_dir_path}")

        # 2. Setup Dataset
        try:
            dataset = Ego4DChunkedTemporalDataset(
                pkl_path=args.pkl_path,
                video_root=args.video_root,
                merge_root=temp_dir_path,
                fps=args.fps,
                chunk_len_sec=CHUNK_LEN_SEC_DEFAULT,
                strict_exists=args.strict_exists,
                only_video_id=args.only_video_id,
            )
        except ValueError as e:
            print(f"\n[Error] Dataset Initialization Failed: {e}")
            return

        total_len = len(dataset)
        if args.max_samples > 0:
            total_len = min(total_len, args.max_samples)
            dataset = torch.utils.data.Subset(dataset, range(total_len))

        print(f"Dataset Size: {total_len}")

        # 3. Setup DataLoader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )

        # 4. Initialize Queue and Progress Bar
        queue = asyncio.Queue(maxsize=args.concurrency * 2)
        results = []

        print(f"Starting processing with {args.concurrency} workers...")
        pbar = tqdm(total=total_len, desc="Processing")

        # 5. Start Producer in a separate Thread
        loop = asyncio.get_running_loop()
        producer_thread = threading.Thread(
            target=dataset_producer, args=(loader, queue, loop), daemon=True
        )
        producer_thread.start()

        # 6. Start Consumers (Workers)
        workers = [
            asyncio.create_task(worker(client, queue, results, pbar))
            for _ in range(args.concurrency)
        ]

        # 7. Wait for workers to finish
        await asyncio.gather(*workers)

        # Close progress bar
        pbar.close()
        producer_thread.join()

    # 8. Post-Processing & Reporting
    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    avg_latency = (
        sum(r["latency_sec"] for r in results) / len(results) if results else 0.0
    )
    success_rate = (len(successes) / len(results) * 100) if results else 0.0

    print("\n" + "=" * 40)
    print("       PROCESSING SUMMARY")
    print("=" * 40)
    print(f"Total Samples  : {len(results)}")
    print(f"Successes      : {len(successes)}")
    print(f"Errors         : {len(errors)}")
    print(f"Success Rate   : {success_rate:.2f}%")
    print(f"Avg Latency    : {avg_latency:.2f} sec/req")
    print("=" * 40)

    if errors:
        print("\nTop 5 Error Examples:")
        for i, err in enumerate(errors[:5]):
            print(f"{i+1}. Vid: {err.get('video_id')} | Err: {err.get('error')}")

    results.sort(key=lambda x: (x.get("video_id", ""), x.get("pred_start_global", 0)))

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
