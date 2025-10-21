import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from second_party.storage.sqlite import SQLiteClient
from second_party.storage.memmap import write_memmap, read_memmap


def main(args):
    assert args.sqlite_path.endswith(".sqlite"), "SQLite file must end with .sqlite"

    print(f"Loading embeddings from {args.sqlite_path}")
    embeddings_matrix = []
    captions = {}

    with SQLiteClient(args.sqlite_path) as client:
        batch_idx = 0
        batch_size = 10_000
        print(f"Total number of embeddings: {client.count_embeddings()}")
        total_batches = client.count_embeddings() // batch_size
        for batch in client.iter_embeddings(batch_size=batch_size):
            print(f"Processing batch {batch_idx} of {total_batches}")
            for i, (caption, embedding) in enumerate(batch):
                captions[caption] = batch_idx * batch_size + i
                embeddings_matrix.append(embedding)

            batch_idx += 1 

    print(f"Loaded {len(embeddings_matrix)} embeddings")
    embeddings_matrix = np.array(embeddings_matrix)

    print(f"Saving embeddings to {Path(args.memap_output_path) / 'embeddings.memmap'}")
    write_memmap(Path(args.memap_output_path) / "embeddings.memmap", embeddings_matrix)

    with open(Path(args.memap_output_path) / "captions.json", "w") as f:
        json.dump(captions, f)

    with open(Path(args.memap_output_path) / "shape.json", "w") as f:
        json.dump({"shape": embeddings_matrix.shape}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", type=str, required=True)
    parser.add_argument("--memap-output-path", type=str, required=True)
    args = parser.parse_args()

    # main(args)
