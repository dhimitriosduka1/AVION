import torch
import pickle as pkl
from collections import defaultdict
from vllm import LLM
import json

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
OUTPUT_DATA_PATH = (
    "/u/dduka/project/AVION/second_party/preprocess/data/merge_candidate_pairs.json"
)
TASK = "Identify the underlying action in this sentence for the purpose of grouping identical events."
SIM_THRESHOLD = 0.85
GPU_COUNT = torch.cuda.device_count()


def get_detailed_instruct(task_description, query):
    return f"Instruct: {task_description}\nQuery:{query}"


def get_model():
    return LLM(model=MODEL_ID, tensor_parallel_size=GPU_COUNT)


if __name__ == "__main__":
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    # Grouping
    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    # Sort segments by start time
    for video_id in grouped_by_video:
        # Each entry is constructed as [uuid, video_id, start_time, end_time, caption]
        grouped_by_video[video_id] = sorted(
            grouped_by_video[video_id], key=lambda x: x[2]
        )
    model = get_model()

    all_pairs = []
    candidate_pairs = []

    for video_id, segments in grouped_by_video.items():
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]

            # If two segments overlap in time and have the same caption, we can directly merge them without inference
            if (
                next_segment[2] <= current_segment[3]
                and current_segment[4] == next_segment[4]
            ):
                all_pairs.append(
                    {
                        "current_segment": current_segment,
                        "next_segment": next_segment,
                        "pair_type": "exact_match",
                        "similarity": 1.0,
                    }
                )
                continue

            # Overlapping segments describing the same action with different captions
            if next_segment[2] <= current_segment[3]:
                candidate_pairs.append((current_segment, next_segment, "overlap"))

    print(f"Total pairs: {len(all_pairs) + len(candidate_pairs)}")
    print(f"Exact match pairs: {len(all_pairs)}")
    print(f"Candidate pairs for similarity check: {len(candidate_pairs)}")

    # Prepare inputs for similarity check
    input_texts = set()
    for current_segment, next_segment, pair_type in candidate_pairs:
        if pair_type == "overlap":
            input_texts.add(get_detailed_instruct(TASK, current_segment[4]))
            input_texts.add(get_detailed_instruct(TASK, next_segment[4]))

    # Convert set to list for consistent ordering
    input_texts = list(input_texts)

    print(f"Unique input texts for embedding: {len(input_texts)}")

    outputs = model.embed(input_texts, use_tqdm=True)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])

    mapping = {text: emb for text, emb in zip(input_texts, embeddings)}

    for current_segment, next_segment, pair_type in candidate_pairs:
        if pair_type == "overlap":
            emb1 = mapping[get_detailed_instruct(TASK, current_segment[4])]
            emb2 = mapping[get_detailed_instruct(TASK, next_segment[4])]
            similarity = (emb1 @ emb2).item()

            if similarity > SIM_THRESHOLD:
                all_pairs.append(
                    {
                        "current_segment": current_segment,
                        "next_segment": next_segment,
                        "pair_type": pair_type,
                        "similarity": similarity,
                    }
                )

    print(f"Total pairs after similarity check: {len(all_pairs)}")

    to_save = []
    for pair in all_pairs:
        to_save.append(
            {
                "uuids": (pair["current_segment"][0], pair["next_segment"][0]),
                "pair_type": pair["pair_type"],
                "similarity": pair["similarity"],
            }
        )

    # Save the results for further analysis
    with open(OUTPUT_DATA_PATH, "w") as f:
        json.dump(to_save, f, indent=4)
