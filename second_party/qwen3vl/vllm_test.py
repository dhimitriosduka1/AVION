import math
import time
import torch
import pickle
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# --- Configuration ---
MODEL_PATH_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK_LEN_SEC_DEFAULT = 15.0
BATCH_SIZE = 256

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
        # uuid, video_id, start, end, caption
        with open(pkl_path, "rb") as f:
            self.rows = pickle.load(f)

        assert len(self.rows[0]) == 5

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

        # Ensure we capture all chunks
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

        # Construct message content
        content = []

        # Add video blocks
        for video_path in paths:
            if not os.path.exists(video_path):
                print(f"Warning: Video missing at {video_path}")

            content.append(
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": self.max_pixels,
                    "fps": self.fps,
                }
            )

        # Add text prompt
        content.append(
            {
                "type": "text",
                "text": PROMPT_TEMPLATE.format(
                    caption=caption, seed_start=rel_start, seed_end=rel_end
                ),
            }
        )

        # Return structured data; message is a LIST of dicts (user role)
        return {
            "uuid": row[0],
            "video_id": row[1],
            "caption": caption,
            "global_start": start,
            "global_end": end,
            "rel_start": rel_start,
            "rel_end": rel_end,
            "chunks": paths,
            "message": [{"role": "user", "content": content}],
        }


# 1. Create the dataset
dataset = Ego4DChunkedTemporalDataset(
    pkl_path="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl",
    video_root="/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/",
    fps=8,
    only_video_id=None,
)

# 2. Initialize the engine
print(f"Initializing vLLM model: {MODEL_PATH_DEFAULT}...")
llm = LLM(
    model=MODEL_PATH_DEFAULT,
    tensor_parallel_size=4,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 5},
)

tokenizer = llm.get_tokenizer()

sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)

# 3. Process items in BATCHES
print(f"Starting batched processing. Batch Size: {BATCH_SIZE}")
total_start = time.time()

# Loop through the dataset in steps of BATCH_SIZE
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch_indices = range(i, min(i + BATCH_SIZE, len(dataset)))

    # Buffers for the current batch
    vllm_inputs_batch = []
    metadata_batch = []

    # --- A. Prepare Batch (CPU Work) ---
    for idx in batch_indices:
        try:
            item = dataset[idx]

            # Apply chat template
            prompt_text = tokenizer.apply_chat_template(
                item["message"], tokenize=False, add_generation_prompt=True
            )

            # Process vision info (loads video bytes from disk)
            image_inputs, video_inputs = process_vision_info(
                item["message"], return_video_metadata=True
            )

            # Construct vLLM input dict
            mm_data = {}
            if image_inputs:
                mm_data["image"] = image_inputs
            if video_inputs:
                mm_data["video"] = video_inputs

            vllm_inputs_batch.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": mm_data,
                }
            )

            metadata_batch.append({"uuid": item["uuid"], "caption": item["caption"]})
        except Exception as e:
            print(f"Error preparing item {idx}: {e}")
            continue

    if not vllm_inputs_batch:
        continue

    # --- B. Run Inference (GPU Work) ---
    print(f"Generating batch {i} to {i+len(vllm_inputs_batch)}...")
    batch_start = time.time()

    # KEY CHANGE: passing a list of inputs allows vLLM to use continuous batching
    outputs = llm.generate(vllm_inputs_batch, sampling_params)

    batch_duration = time.time() - batch_start

    # # --- C. Process Results ---
    for j, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = metadata_batch[j]

        print(f"[UUID: {meta['uuid']}] Caption: {meta['caption']}")
        print(f"Output: {generated_text}")
        print("-" * 30)

    print(
        f"Batch finished in {batch_duration:.2f}s (Avg {batch_duration/len(vllm_inputs_batch):.2f}s per item)"
    )

print(f"Total execution time: {time.time() - total_start:.2f}s")
