import math
import time
import torch
import pickle
import os
import io

from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# --- Configuration ---
MODEL_PATH_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK_LEN_SEC_DEFAULT = 15.0
BATCH_SIZE = 16
FPS = 8
MAX_PIXELS = 360 * 420

PKL_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
VIDEO_ROOT = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/"

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
        self, pkl_path, video_root, fps, only_video_id=None, max_pixels=360 * 420
    ):
        self.video_root = video_root
        self.fps = int(fps)
        self.chunk_len_sec = CHUNK_LEN_SEC_DEFAULT
        self.max_pixels = max_pixels

        print(f"Loading dataset from {pkl_path}...")
        # Expects: uuid, video_id, start, end, caption
        with open(pkl_path, "rb") as f:
            self.rows = pickle.load(f)

        if only_video_id is not None:
            self.rows = [r for r in self.rows if r[1] == only_video_id]
        print(f"Dataset loaded with {len(self.rows)} items.")

    def _get_chunk_path(self, root, video_id, chunk_id):
        return os.path.join(root, f"{video_id}.mp4", f"{chunk_id}.mp4")

    def _chunk_id_from_time(self, t):
        return int(math.floor(t / self.chunk_len_sec) * self.chunk_len_sec)

    def _get_covering_chunk_ids(self, start, end):
        first_chunk = self._chunk_id_from_time(start)
        last_chunk = self._chunk_id_from_time(end)

        chunks = []
        curr = first_chunk

        # Ensure we capture all chunks covering the time range
        while curr <= last_chunk:
            chunks.append(curr)
            curr += int(self.chunk_len_sec)

        return chunks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        uuid, video_id, start, end, caption = row

        chunk_ids = self._get_covering_chunk_ids(start, end)

        paths = []
        for cid in chunk_ids:
            c_path = self._get_chunk_path(self.video_root, video_id, cid)
            paths.append(c_path)

        base_offset = float(chunk_ids[0])
        rel_start = start - base_offset
        rel_end = end - base_offset

        # Return metadata needed to construct the prompt and load videos later
        return {
            "uuid": uuid,
            "video_id": video_id,
            "caption": caption,
            "chunks": paths,  # List of file paths
            "global_start": start,
            "global_end": end,
            "text_prompt": PROMPT_TEMPLATE.format(
                caption=caption, seed_start=rel_start, seed_end=rel_end
            ),
        }


def load_chunk_tensor(video_path, fps, max_pixels):
    """
    Reads a video file and processes it into the tensor format vLLM expects.
    Returns: A list of tensors (usually length 1 for a single video file).
    """
    if not os.path.exists(video_path):
        # Return None so we can handle missing files gracefully
        return None

    # We construct a minimal "message" just to use qwen_vl_utils for loading
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
        # process_vision_info returns (image_inputs, video_inputs)
        # We only care about video_inputs here.
        _, video_inputs = process_vision_info(dummy_message, return_video_metadata=True)
        return video_inputs
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None


# --- Main Execution ---

# 1. Initialize Dataset
dataset = Ego4DChunkedTemporalDataset(
    pkl_path=PKL_PATH,
    video_root=VIDEO_ROOT,
    fps=FPS,
    max_pixels=MAX_PIXELS,
    only_video_id=None,
)

# 2. Initialize vLLM Engine
print(f"Initializing vLLM model: {MODEL_PATH_DEFAULT}...")
llm = LLM(
    model=MODEL_PATH_DEFAULT,
    tensor_parallel_size=4,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 10},  # Allow enough chunks per prompt
)

tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)

# 3. Process Batches
print(f"Starting batched processing. Batch Size: {BATCH_SIZE}")
total_start = time.time()

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch_indices = range(i, min(i + BATCH_SIZE, len(dataset)))

    # Get raw items from dataset (metadata only, no video loaded yet)
    batch_items = [dataset[idx] for idx in batch_indices]

    # --- Step A: Identify Unique Videos ---
    unique_paths = set()
    for item in batch_items:
        unique_paths.update(item["chunks"])

    # --- Step B: Load Videos into Batch Cache ---
    # This ensures every file is read from disk exactly once per batch
    chunk_cache = {}  # path -> video_tensor

    for path in unique_paths:
        tensor_output = load_chunk_tensor(path, fps=FPS, max_pixels=MAX_PIXELS)
        if tensor_output is not None:
            chunk_cache[path] = tensor_output

    # --- Step C: Build vLLM Inputs ---
    vllm_inputs_batch = []
    metadata_batch = []

    for item in batch_items:
        content_list = []
        combined_video_tensors = []

        # Add video blocks
        for path in item["chunks"]:
            if path in chunk_cache:
                # 1. Add placeholder for prompt text structure
                # We provide a dummy "video" key so the template processor knows a video exists here
                content_list.append({"type": "video", "video": "placeholder"})

                # 2. Collect the actual tensor data
                combined_video_tensors.extend(chunk_cache[path])
            else:
                # Handle missing video gracefully (warn and skip adding)
                print(
                    f"Warning: Skipping missing video chunk for UUID {item['uuid']}: {path}"
                )

        # Add text instruction
        content_list.append({"type": "text", "text": item["text_prompt"]})

        # Apply Chat Template
        conversation = [{"role": "user", "content": content_list}]
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Only add to batch if we successfully loaded at least one video
        if combined_video_tensors:
            vllm_inputs_batch.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": {"video": combined_video_tensors},
                }
            )
            metadata_batch.append(item)

    if not vllm_inputs_batch:
        continue

    # --- Step D: Run Inference ---
    print(f"Generating batch {i} to {i + len(vllm_inputs_batch)}...")
    batch_start = time.time()

    try:
        outputs = llm.generate(vllm_inputs_batch, sampling_params)
    except Exception as e:
        print(f"Inference error in batch {i}: {e}")
        continue

    batch_duration = time.time() - batch_start

    # --- Step E: Handle Results ---
    for j, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = metadata_batch[j]

        print(
            f"[UUID: {meta['uuid']}, VIDEO ID: {meta['video_id']}, GLOABL START: {meta['global_start']} GLOBAL END: {meta['global_end']}] Caption: {meta['caption']}"
        )
        print(f"Output: {generated_text}")
        print("-" * 30)

    if i > 100:
        break


print(f"Total execution time: {time.time() - total_start:.2f}s")
