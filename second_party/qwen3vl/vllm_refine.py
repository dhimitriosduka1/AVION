import os
import cv2
import math
import time
import torch
import pickle
import argparse
import json
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# --- Default Configuration ---
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHUNK_LEN = 15.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_FPS = 8
DEFAULT_MAX_PIXELS = 360 * 420
DEFAULT_PKL_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
DEFAULT_VIDEO_ROOT = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/"
DEFAULT_PADDING = 0
DEFAULT_OUTPUT_FILE_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/output.jsonl"
)
DEFAULT_VIDEO_LEN_PATH = "/dais/fs/scratch/dduka/databases/ego4d/video_lengths.json"

PROMPT_TEMPLATE = """
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
    END
"""


class Ego4DChunkedTemporalDataset(torch.utils.data.Dataset):
    def __init__(
        self, pkl_path, video_root, fps, chunk_len_sec, padding, only_video_id=None
    ):
        self.video_root = video_root
        self.fps = int(fps)
        self.chunk_len_sec = chunk_len_sec
        self.padding = padding
        self.nr_samples_processed = 0

        print(f"Loading dataset from {pkl_path}...")
        # Expects: uuid, video_id, start, end, caption
        with open(pkl_path, "rb") as f:
            self.all_rows = pickle.load(f)

        if only_video_id is not None:
            self.all_rows = [r for r in self.all_rows if r[1] == only_video_id]

        print(f"Dataset loaded with {len(self.all_rows)} total items.")

        self._compute_video_lengths()

    def _compute_video_lengths(self):
        if DEFAULT_VIDEO_LEN_PATH != "" and os.path.exists(DEFAULT_VIDEO_LEN_PATH):
            print("Loading from saved video lengths file...")
            with open(DEFAULT_VIDEO_LEN_PATH, "r") as f:
                self.video_lengths = json.load(f)
                return

        self.video_lengths = {}
        for current_dir, _, files in tqdm(
            os.walk(self.video_root), desc="Computing video lengths..."
        ):
            chunks = [f for f in files if f.endswith(".mp4")]
            if not chunks:
                continue

            video_id = os.path.basename(current_dir)

            try:
                last_chunk = max(chunks, key=lambda x: int(os.path.splitext(x)[0]))
                last_path = os.path.join(current_dir, last_chunk)
                duration = self._get_video_duration(last_path)

                last_chunk_start_time = float(os.path.splitext(last_chunk)[0])
                self.video_lengths[video_id] = last_chunk_start_time + duration
            except ValueError:
                continue

    def _get_video_duration(self, file_path):
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return 0.0

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps > 0:
                duration = frame_count / fps
            else:
                duration = 0.0

            cap.release()
            return duration
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0.0

    def _get_chunk_path(self, root, video_id, chunk_id):
        return os.path.join(root, f"{video_id}.mp4", f"{chunk_id}.mp4")

    def _chunk_id_from_time(self, t):
        return int(math.floor(t / self.chunk_len_sec) * self.chunk_len_sec)

    def _get_covering_chunk_ids(self, video_id, start, end):
        first_chunk = self._chunk_id_from_time(start)
        last_chunk = self._chunk_id_from_time(end)

        if self.nr_samples_processed < 100:
            print(
                f"Before padding first_chunk: {first_chunk}, last_chunk: {last_chunk}"
            )

        if self.padding != 0:
            first_chunk = max(0, first_chunk - self.chunk_len_sec * self.padding)
            last_chunk = min(
                last_chunk + self.chunk_len_sec * self.padding,
                self._chunk_id_from_time(self.video_lengths[video_id]),
            )

        if self.nr_samples_processed < 100:
            print(f"After padding first_chunk: {first_chunk}, last_chunk: {last_chunk}")

        chunks = []
        curr = first_chunk
        while curr <= last_chunk:
            chunks.append(int(curr))
            curr += self.chunk_len_sec

        return chunks

    def __len__(self):
        return len(self.all_rows)

    def __getitem__(self, idx):
        row = self.all_rows[idx]
        uuid, video_id, start, end, caption = row

        if video_id not in self.video_lengths:
            print(f"Skipping video_id: {video_id}")
            return None

        chunk_ids = self._get_covering_chunk_ids(video_id, start, end)
        if chunk_ids is None or len(chunk_ids) == 0:
            print(
                f"No chunks found for VIDEO ID: {video_id} with START: {start} and END: {end} with LEN: {self.video_lengths[video_id]}"
            )
            return None

        paths = []
        for cid in chunk_ids:
            c_path = self._get_chunk_path(self.video_root, video_id, cid)
            paths.append(c_path)

        base_offset = float(chunk_ids[0])

        rel_start = start - base_offset
        rel_end = end - base_offset

        # Increment the number of samples processed
        self.nr_samples_processed += 1

        return {
            "uuid": uuid,
            "video_id": video_id,
            "caption": caption,
            "chunks": paths,
            "global_start": start,
            "global_end": end,
            "rel_start": rel_start,
            "rel_end": rel_end,
            "video_length": self.video_lengths[video_id],
            "base_offset": base_offset,
            "text_prompt": PROMPT_TEMPLATE.format(
                caption=caption, seed_start=rel_start, seed_end=rel_end
            ),
        }


def load_chunk_tensor(video_path, fps, max_pixels):
    """
    Reads a video file and processes it into the tensor format vLLM expects.
    """
    if not os.path.exists(video_path):
        return None

    dummy_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": max_pixels,
                    "fps": fps,
                }
            ],
        }
    ]

    try:
        _, video_inputs = process_vision_info(dummy_message, return_video_metadata=True)
        return video_inputs
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen-VL inference on Ego4D dataset chunks."
    )

    # Model and Hardware Config
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )

    # Data Config
    parser.add_argument(
        "--pkl_path",
        type=str,
        default=DEFAULT_PKL_PATH,
        help="Path to the dataset pickle file.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=DEFAULT_VIDEO_ROOT,
        help="Root directory for video chunks.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_PATH,
        help="Optional path to save output (will default to .jsonl extension).",
    )

    # Video Processing Config
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help="FPS to sample videos at."
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Max pixels for Qwen-VL processing.",
    )
    parser.add_argument(
        "--chunk_len_sec",
        type=float,
        default=DEFAULT_CHUNK_LEN,
        help="Length of each video chunk in seconds.",
    )
    parser.add_argument(
        "--video_padding",
        type=int,
        default=DEFAULT_PADDING,
        help="Number of clips to pad when retrieving a sample from the dataset",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of the dataset to process (inclusive).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="End index of the dataset to process (exclusive). Use -1 for end of dataset.",
    )

    args = parser.parse_args()

    dataset = Ego4DChunkedTemporalDataset(
        pkl_path=args.pkl_path,
        video_root=args.video_root,
        fps=args.fps,
        chunk_len_sec=args.chunk_len_sec,
        padding=args.video_padding,
        only_video_id=None,
    )

    total_len = len(dataset)
    start_idx = args.start_idx
    if args.end_idx == -1 or args.end_idx > total_len:
        end_idx = total_len
    else:
        end_idx = args.end_idx

    if start_idx >= end_idx:
        print(
            f"Start index ({start_idx}) >= End index ({end_idx}). Nothing to process."
        )
        return

    print(
        f"Processing range: [{start_idx}, {end_idx}) (Total items: {end_idx - start_idx})"
    )
    print(f"Video Padding: {args.video_padding}s")

    final_output_path = None
    if args.output_file:
        original_path = Path(args.output_file)
        # Force .jsonl extension for clarity and safety
        new_filename = f"{original_path.stem}_{start_idx}_{end_idx}.jsonl"
        final_output_path = original_path.parent / new_filename

        # Ensure directory exists
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving output to: {final_output_path}")

    print(f"Initializing vLLM model: {args.model_path}...")

    mm_processor_kwargs = {"mm_encoder_tp_mode": "data"}

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 10},
        mm_processor_kwargs=mm_processor_kwargs,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.05,
    )

    print(f"Starting batched processing. Batch Size: {args.batch_size}")
    total_start_time = time.time()

    # Load already processed UUIDs from existing output file
    processed_uuids = set()
    if final_output_path and final_output_path.exists():
        try:
            with open(final_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        processed_uuids.add(entry["uuid"])
            print(
                f"Found {len(processed_uuids)} already processed entries in output file."
            )
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    # Check if all items are already processed
    if len(processed_uuids) == end_idx - start_idx:
        print(
            f"All captions are processed ({len(processed_uuids)} entries), exiting..."
        )
        exit(0)

    # Filter out the already computed indices
    indices_to_process = []
    for idx in range(start_idx, end_idx):
        item = dataset[idx]
        if item is not None and item["uuid"] not in processed_uuids:
            indices_to_process.append(idx)

    print(
        f"Indices to process after filtering: {len(indices_to_process)} (skipped {end_idx - start_idx - len(indices_to_process)} already processed)"
    )

    out_f = None
    if final_output_path:
        out_f = open(final_output_path, "a", encoding="utf-8")

    current_idx = 0  # Now indexes into indices_to_process
    pbar = tqdm(total=len(indices_to_process))

    try:
        while current_idx < len(indices_to_process):
            batch_end = min(current_idx + args.batch_size, len(indices_to_process))
            batch_indices = indices_to_process[current_idx:batch_end]

            batch_items_raw = [dataset[idx] for idx in batch_indices]

            valid_batch_items = [item for item in batch_items_raw if item is not None]

            if not valid_batch_items:
                pbar.update(len(batch_indices))
                current_idx = batch_end
                continue

            # --- 1. Cache Video Tensors for this Batch ---
            unique_paths = set()
            for item in valid_batch_items:
                unique_paths.update(item["chunks"])

            chunk_cache = {}
            for path in unique_paths:
                if path not in chunk_cache:
                    tensor_output = load_chunk_tensor(
                        path, fps=args.fps, max_pixels=args.max_pixels
                    )
                    if tensor_output is not None:
                        chunk_cache[path] = tensor_output

            # --- 2. Construct vLLM Inputs ---
            vllm_inputs_batch = []
            metadata_batch = []

            for item in valid_batch_items:
                content_list = []
                combined_video_tensors = []
                valid_item = True

                for path in item["chunks"]:
                    if path in chunk_cache:
                        content_list.append({"type": "video", "video": "placeholder"})
                        combined_video_tensors.extend(chunk_cache[path])
                    else:
                        print(
                            f"Warning: Skipping missing video chunk for UUID {item['uuid']}: {path}"
                        )
                        valid_item = False
                        break

                if valid_item:
                    content_list.append({"type": "text", "text": item["text_prompt"]})
                    conversation = [{"role": "user", "content": content_list}]
                    prompt_text = tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )

                    vllm_inputs_batch.append(
                        {
                            "prompt": prompt_text,
                            "multi_modal_data": {"video": combined_video_tensors},
                        }
                    )
                    metadata_batch.append(item)

            # --- 3. Generate ---
            if vllm_inputs_batch:
                try:
                    outputs = llm.generate(
                        vllm_inputs_batch, sampling_params, use_tqdm=False
                    )

                    for j, output in enumerate(outputs):
                        generated_text = output.outputs[0].text
                        meta = metadata_batch[j]

                        # --- 4. Robust JSON Parsing ---
                        try:
                            clean_text = (
                                generated_text.replace("```json", "")
                                .replace("```", "")
                                .strip()
                            )
                            model_json_output = json.loads(clean_text)
                        except json.JSONDecodeError:
                            model_json_output = {
                                "raw_output": generated_text,
                                "error": "Model output not valid JSON",
                            }

                        result_entry = {
                            "uuid": meta["uuid"],
                            "video_id": meta["video_id"],
                            "rel_start": meta["rel_start"],
                            "rel_end": meta["rel_end"],
                            "global_start": meta["global_start"],
                            "global_end": meta["global_end"],
                            "caption": meta["caption"],
                            "base_offset": meta["base_offset"],
                            "padding_used": args.video_padding,
                            "model_output": model_json_output,
                        }

                        if out_f:
                            json_line = json.dumps(result_entry, ensure_ascii=False)
                            out_f.write(json_line + "\n")
                            out_f.flush()
                        else:
                            print(f"[UUID: {meta['uuid']}] Output: {generated_text}")

                except Exception as e:
                    print(f"Inference error in batch starting at {current_idx}: {e}")
                    import traceback

                    traceback.print_exc()

            pbar.update(len(batch_indices))
            current_idx = batch_end

    finally:
        if out_f:
            out_f.close()
            print("File closed successfully.")
        pbar.close()

    print(f"Job finished. Execution time: {time.time() - total_start_time:.2f}s")


if __name__ == "__main__":
    main()
