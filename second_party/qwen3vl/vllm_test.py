import math 
import time
import torch
import pickle

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"
CHUNK_LEN_SEC_DEFAULT = 15.0

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
        self,
        pkl_path,
        video_root,
        fps,
        only_video_id
    ):
        self.video_root = video_root
        self.fps = int(fps)
        self.chunk_len_sec = CHUNK_LEN_SEC_DEFAULT

        with open(pkl_path, "rb") as f:
            self.rows = pickle.load(f)

        assert len(rows[0]) == 5

        if only_video_id not None:
            self.rows = [r for r in self.rows if r[1] == only_video_id]
    
    def _chunk_id_from_time(self, t):
        return int(math.floor(t / self.chunk_len_sec) * self.chunk_len_sec)

    def _get_covering_chunk_ids(self, start, end):
        first_chunk = self._chunk_id_from_time(start)
        last_chunk = self._chunk_id_from_time(end)

        chunks = []
        curr = first_chunk

        while curr <= last_chunk:
            chunks.append(curr)
            curr += int(self.chunk_len_sec)

        return chunks

    def __len__(self):
        return len(self.rows)


    def __getitem__(self, idx):
        row = self.rows[idx]
        start = row[2]
        end = row[3]

        chunk_ids = self._get_covering_chunk_ids(start, end)

        paths = []
        for cid in chunk_ids:
            chunk_path = chunk_path(self.video_root, video_id, cid)
            paths.append(chunk_path)
        
        base_offset = float(chunk_ids[0])
        rel_start = start - base_offset
        rel_end = end - base_offset

        content = [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": self.fps,
            } for video_path in paths
        ]

        content.append(
            {
                "type": "text", "text": PROMPT_TEMPLATE.format(
                    caption=row[-1], seed_start=rel_start, seed_end=rel_end
                )
            }
        )

        return {
            "uuid": row[0],
            "video_id": row[1],
            "caption": row[-1]
            "global_start": start,
            "global_end": end,
            "rel_start": rel_start,
            "rel_end": rel_end,
            "chunks": paths,
            "message": {
                "role": "user",
                "content": content
            },
        }


# Create the dataset
dataset = Ego4DChunkedTemporalDataset(
    pkl_path="",
    video_root="",
    fps=8,
    only_video_id=""
)

# 1. Initialize the engine
llm = LLM(
    model=MODEL_PATH_DEFAULT,
    tensor_parallel_size=4,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 1},
)

# 2. Prepare the video path
video_path = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/f62d3060-470f-49b3-9f32-8239a519908c.mp4/15.mp4"

# 3. Define the conversation
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": 8.0,
            },
            {"type": "text", "text": "Describe what's happening in this video."},
        ],
    }
]

# 4. Apply chat template
tokenizer = llm.get_tokenizer()
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 5. Process vision info using Qwen utils
image_inputs, video_inputs = process_vision_info(messages, return_video_metadata=True)

# 6. Construct vLLM input with proper multi_modal_data structure
# The key fix: ensure video_inputs contains both frames and metadata
mm_data = {}
if image_inputs:
    mm_data["image"] = image_inputs
if video_inputs:
    mm_data["video"] = video_inputs

vllm_inputs = {
    "prompt": prompt_text,
    "multi_modal_data": mm_data,
}

# 7. Generate
sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

start = time.time()
outputs = llm.generate([vllm_inputs], sampling_params)
duration = time.time() - start

for output in outputs:
    print(f"Response costs: {duration:.2f}s")
    print(f"Generated text: {output.outputs[0].text}")
