import os
import re
import json
import torch
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
DEDUPLICATED_DATA_PATH = (
    "/ptmp/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
OUTPUT_DATA_PATH = (
    "/ptmp/dduka/databases/ego4d/ego4d_train_deduplicated_and_llm_merged_with_uuid.pkl"
)
VIDEO_ROOT = "/ptmp/dduka/databases/ego4d/video_320px_15sec/"

GPU_COUNT = torch.cuda.device_count()
PROMPT_TEMPLATE = """"""


def generate_merge_candidates(samples_by_video_id):
    print("Generating merge candidates...")

    dataset = []
    merge_candidates = []
    for video_id, samples in tqdm(samples_by_video_id.items()):
        samples.sort(key=lambda x: x[2])

        if not samples:
            continue

        current_merged = list(samples[0])
        history = [samples[0]]

        for i in range(1, len(samples)):
            next_sample = samples[i]

            if next_sample[2] <= current_merged[3]:
                current_merged[3] = max(current_merged[3], next_sample[3])
                history.append(next_sample)
            else:
                if len(history) > 1:
                    merge_candidates.append(
                        {
                            "video_id": video_id,
                            "history": list(history),
                            "merged_row": list(current_merged),
                        }
                    )
                else:
                    dataset.append(tuple(current_merged))

                current_merged, history = list(next_sample), [next_sample]

        # Handle the final group
        if len(history) > 1:
            merge_candidates.append(
                {
                    "video_id": video_id,
                    "history": list(history),
                    "merged_row": list(current_merged),
                }
            )
        else:
            dataset.append(tuple(current_merged))

    if merge_candidates:
        print(
            f"Found {len(merge_candidates)} potential merges. {sum(len(s['history']) for s in merge_candidates)} samples involved."
        )
        print(
            f"Average merge candidate has {sum(len(s['history']) for s in merge_candidates) / len(merge_candidates):.2f} samples."
        )
    else:
        print("No merge candidates found.")

    return dataset, merge_candidates


def load_model():
    llm = LLM(
        model=DEFAULT_MODEL_PATH,
        tensor_parallel_size=GPU_COUNT,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, top_k=20, max_tokens=2048
    )
    return llm, tokenizer, sampling_params


if __name__ == "__main__":
    print(f"Using {GPU_COUNT} GPUs")

    llm, tokenizer, sampling_params = load_model()

    if os.path.exists(DEDUPLICATED_DATA_PATH):
        with open(DEDUPLICATED_DATA_PATH, "rb") as f:
            dataset = pkl.load(f)
            print(f"Loaded {len(dataset)} samples.")
    else:
        print(f"Path not found: {DEDUPLICATED_DATA_PATH}")
        dataset = []

    samples_by_video_id = defaultdict(list)
    for sample in dataset:
        samples_by_video_id[sample[1]].append(sample)

    dataset, merge_candidates = generate_merge_candidates(samples_by_video_id)

    # Put your prompts in a list
    prompts = [
        "Compare these two egocentric video captions:\n1: '#C C shapes an origami'\n2: '#C C picks-up a glue from the table with her right hand.'\nDo they describe the same action or state? Answer with YES or No.",
        "Compare these two egocentric video captions:\n1: '#C C shapes an origami'\n2: '#C C makes paper craft on the table in a room.'\nDo they describe the same action or state? Answer with YES or No.",
    ]

    task_batches = [[{"role": "user", "content": p}] for p in prompts]

    outputs = llm.chat(
        task_batches,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": True},
    )

    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text

        clean_text = re.sub(
            r"<think>.*?</think>", "", raw_text, flags=re.DOTALL
        ).strip()

        print(f"\n--- Result for Task {i+1} ---")
        try:
            # Try to parse the clean text as JSON
            data = json.loads(clean_text)
            print(f"Merge Decision: {data['merge']}")
            print(f"Confidence: {data['confidence']}")
        except json.JSONDecodeError:
            # If parsing fails, print the raw text
            print(f"Raw Output: {raw_text}")

    # prompt = "Compare these two egocentric video captions:\n1: '#C C shapes an origami'\n2: '#C C picks-up a glue from the table with her right hand.'\n Do they describe the same action or state? Answer with a JSON object with two fields: 'merge' (boolean) indicating whether to merge the captions, and 'confidence' (float between 0 and 1) indicating your confidence level in the decision."
    # prompt2 = "Compare these two egocentric video captions:\n1: '#C C shapes an origami'\n2: '#C C makes paper craft on the table in a room.'\n Do they describe the same action or state? Answer with a JSON object with two fields: 'merge' (boolean) indicating whether to merge the captions, and 'confidence' (float between 0 and 1) indicating your confidence level in the decision."
    # messages = [
    #     {"role": "user", "content": prompt},
    #     {"role": "assistant", "content": prompt2},
    # ]

    # # Generate outputs
    # outputs = llm.chat(
    #     [messages],
    #     sampling_params=sampling_params,
    # )

    # for output in outputs:
    #     # prompt = output.prompt # Already defined in your loop
    #     generated_text = output.outputs[0].text

    #     print("-" * 30)
    #     print(f"PROMPT: {output.prompt}")
    #     print(f"RESPONSE:\n{generated_text}")

    # for output in outputs:

    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# def prepare_prompt(candidates, tokenizer):
#     """
#     Prepare prompts for vLLM generate() API with multi-modal data.
#     Uses process_vision_info to handle video processing.
#     """
#     batch_inputs = []
#     for candidate in candidates:
#         chunk_paths, first_chunk = get_video_chunks_for_segments(
#             candidate["video_id"],
#             candidate["merged_row"][2],
#             candidate["merged_row"][3],
#         )

#         segments_as_str = []
#         for idx, history_item in enumerate(candidate["history"]):
#             segments_as_str.append(
#                 f'- {idx + 1}. [{(history_item[2] - first_chunk):.2f}s - {(history_item[3] - first_chunk):.2f}s]: "{history_item[4]}"'
#             )

#         segments_formatted = "\n".join(segments_as_str)
#         prompt_filled = PROMPT_TEMPLATE.format(segments=segments_formatted)

#         # Build messages in Qwen VL chat format
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     *[
#                         {"type": "video", "video": path, "fps": FPS}
#                         for path in chunk_paths
#                     ],
#                     {"type": "text", "text": prompt_filled},
#                 ],
#             }
#         ]

#         # Apply chat template to get the prompt text
#         prompt = tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         # Use process_vision_info to extract and process video data
#         _, video_inputs = process_vision_info(messages, return_video_metadata=True)

#         batch_inputs.append(
#             {
#                 "prompt": prompt,
#                 "multi_modal_data": {"video": video_inputs},
#             }
#         )

#     return batch_inputs


# def chunk_list(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]


# if __name__ == "__main__":
#     print(f"Using {GPU_COUNT} GPUs")

#     llm, tokenizer, sampling_params = load_model()

#     if os.path.exists(DEDUPLICATED_DATA_PATH):
#         with open(DEDUPLICATED_DATA_PATH, "rb") as f:
#             dataset = pkl.load(f)
#             print(f"Loaded {len(dataset)} samples.")
#     else:
#         print(f"Path not found: {DEDUPLICATED_DATA_PATH}")
#         dataset = []

#     samples_by_video_id = defaultdict(list)
#     for sample in dataset:
#         samples_by_video_id[sample[1]].append(sample)


#     dataset, merge_candidates, auto_merged_count = generate_merge_candidates(
#         samples_by_video_id
#     )

#     merged_count = 0
#     for candidates in tqdm(chunk_list(merge_candidates, MINI_BATCH_SIZE)):
#         batch_inputs = prepare_prompt(candidates, tokenizer)

#         outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

#         for candidate, output in zip(candidates, outputs):
#             response_text = output.outputs[0].text.strip()

#             try:
#                 think_end_idx = response_text.index(THINK_KEYWORD) + len(THINK_KEYWORD)

#                 thinking_section = response_text[:think_end_idx].strip()
#                 response_text = response_text[think_end_idx:].strip()

#                 json_response = json.loads(response_text)

#                 do_merge = json_response["merge"]
#                 confidence = json_response["confidence"]

#                 if do_merge and confidence >= 0.9:
#                     print(f"[INFO] Segments must be merged!")
#                     dataset.append(candidate["merged_row"])
#                     merged_count += (
#                         len(candidate["history"]) - 1
#                     )  # -1 since I need to account for the fact that I'm keeping one from all of them
#                 elif do_merge and confidence < 0.9:
#                     print(
#                         f"[Info] The model was not confident in its answer! Keeping the original captions!"
#                     )
#                     dataset.extend(candidate["history"])
#                     print(
#                         f"[INFO] Model not confident enough! Keeping the old captions!"
#                     )
#                     print(f"[INFO] Candidates: {candidate}")
#                 else:
#                     print(
#                         f"[INFO] Segments must not be merged! Keeping the old captions!"
#                     )
#                     print(f"[INFO] Candidates: {candidate}")
#                     dataset.extend(candidate["history"])

#             except (json.JSONDecodeError, Exception) as e:
#                 print(
#                     f"[ERROR] Failed to parse output for {candidate['video_id']}: {e}"
#                 )
#                 print(f"[INFO] Candidates: {candidate}")
#                 dataset.extend(candidate["history"])


# # Write the captions in a file
# with open(OUTPUT_DATA_PATH, "wb") as f:
#     pkl.dump(dataset, f)

# total_merged = merged_count + auto_merged_count

# print(f"Original deduplicated dataset size: {dedup_size}")
# print(f"Number of samples merged by LLM: {merged_count}")
# print(f"Number of sample auto-merged: {auto_merged_count}")
# print(f"Total samples merged: {total_merged}")

# print(f"Expected number of samples to be saved: {dedup_size - total_merged}")
# print(f"Saved {len(dataset)} samples in {OUTPUT_DATA_PATH}")
