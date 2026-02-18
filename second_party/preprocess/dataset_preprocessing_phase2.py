import json
import torch
import pickle as pkl
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# NOTE: In case I need to run this script again, I need to optimize it!

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
TASK = "Identify the underlying action in this sentence for the purpose of grouping identical events."


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    return model, tokenizer, device


def get_detailed_instruct(task_description, query):
    return f"Instruct: {task_description}\nQuery:{query}"


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

    # Load the model
    model, tokenizer, device = load_model_and_tokenizer(MODEL_ID)

    uuids_to_merge = []

    for video_id, segments in tqdm(grouped_by_video.items(), desc="Processing videos"):
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]

            current_caption = current_segment[4]
            next_caption = next_segment[4]

            if current_caption == next_caption:
                continue

            if next_segment[2] <= current_segment[3]:
                input_texts = [
                    get_detailed_instruct(TASK, current_caption),
                    get_detailed_instruct(TASK, next_caption),
                ]

                batch = tokenizer(
                    input_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    outputs = model(**batch)
                    emb = last_token_pool(
                        outputs.last_hidden_state, batch["attention_mask"]
                    )
                    emb = F.normalize(emb, p=2, dim=1)

                similarity = (emb[0] @ emb[1]).item()
                print(
                    f"Similarity: {similarity:.4f} | Pair: ('{current_caption}', '{next_caption}')"
                )

                if similarity > 0.9:
                    uuids_to_merge.append((current_segment[0], next_segment[0]))

    print(f"Total pairs to merge: {len(uuids_to_merge)}")

    with open(
        "/u/dduka/project/AVION/second_party/preprocess/data/uuids_to_merge_phase_2.json",
        "w",
    ) as f:
        json.dump(uuids_to_merge, f, indent=4)
