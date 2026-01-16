import time
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# 1. Initialize the engine
llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
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
