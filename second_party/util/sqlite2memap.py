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

    captions = {}

    with SQLiteClient(args.sqlite_path) as client:
        total_embeddings = client.count_embeddings()
        shape = (total_embeddings, 1280)
        print(f"Shape: {shape}")
        memmap = np.memmap(
            Path(args.memap_output_path) / "embeddings.memmap", shape=shape, mode="w+"
        )

        batch_idx = 0
        batch_size = 10_000

        print(f"Total number of embeddings: {total_embeddings}")
        total_batches = total_embeddings // batch_size

        for batch in client.iter_embeddings(batch_size=batch_size):
            print(f"Processing batch {batch_idx + 1} of {total_batches}")
            for i, (caption, embedding) in enumerate(batch):
                captions[caption] = batch_idx * batch_size + i
                memmap[batch_idx * batch_size + i] = embedding

            batch_idx += 1

            if batch_idx % 100 == 0:
                memmap.flush()

        memmap.flush()

    with open(Path(args.memap_output_path) / "captions.json", "w") as f:
        json.dump(captions, f)

    with open(Path(args.memap_output_path) / "shape.json", "w") as f:
        json.dump({"shape": shape}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", type=str, required=True)
    parser.add_argument("--memap-output-path", type=str, required=True)
    args = parser.parse_args()

    print(args)
    main(args)
