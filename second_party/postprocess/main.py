import os
import json
import math
import torch
import pickle
import argparse

from tqdm import tqdm

from second_party.storage.sqlite import SQLiteClient
from second_party.preprocess.utils import preprocess_captions


def resolve_video_chunk_path(video_root, video_id, start, end, chunk_size=15):
    """
    Return the list of chunk file paths that together cover the timeline [start, end).
    Chunks are 15s long: [0,15), [15,30), [30,45), ...

    Paths are formatted as:
      <video_root>/<video_id>/<index>.mp4

    Examples:
      - start=2.3, end=5.4  -> chunks [0]
      - start=10.3, end=16.9 -> chunks [0, 1]  (crosses the 15s boundary)
    """
    if start < 0 or end < 0:
        raise ValueError("Start and end must be non-negative")

    if end <= start:
        raise ValueError("End must be greater than start")

    epsilon = 1e-9
    end_adj = max(0.0, end - epsilon)

    start_idx = int(math.floor(start / chunk_size))
    end_idx = int(math.floor(end_adj / chunk_size))

    paths = []
    for idx in range(start_idx, end_idx + 1):
        chunk_start_time = int(idx * chunk_size)
        chunk_filename = f"{chunk_start_time}.mp4"
        paths.append(os.path.join(str(video_root), str(video_id), chunk_filename))

    return paths


def get_chunks_metadata(paths):
    return [json.load(open(path)) for path in paths]


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"
    assert args.ego4d_embeddings_path.endswith(
        ".sqlite"
    ), "Ego4d embeddings must be a sqlite file"
    assert args.lavila_embeddings_path.endswith(
        ".sqlite"
    ), "LaViLa embeddings must be a sqlite file"

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    print(f"Opening {args.ego4d_embeddings_path} embeddings")
    ego4d_embeddings = SQLiteClient(args.ego4d_embeddings_path)
    print(f"Loaded {ego4d_embeddings.count_embeddings()} ego4d embeddings")

    print(f"Opening {args.lavila_embeddings_path} embeddings")
    lavila_embeddings = SQLiteClient(args.lavila_embeddings_path)
    print(f"Loaded {lavila_embeddings.count_embeddings()} lavila embeddings")

    for sample in tqdm(data, desc="Processing samples"):
        video_id, start, end, caption = sample
        caption = preprocess_captions([caption])[0]
        caption_embedding = ego4d_embeddings.get_embedding(caption)

        # The anchor caption against which the similarities are computed.
        anchor_caption = torch.from_numpy(caption_embedding)

        video_chunk_paths = resolve_video_chunk_path(
            args.video_root, video_id, start, end, args.chunk_size
        )

        # video_chunk_metadata = get_chunks_metadata(video_chunk_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ego4d-embeddings-path", type=str, required=True)
    parser.add_argument("--lavila-embeddings-path", type=str, required=True)
    parser.add_argument("--chunk-metadata-root", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
