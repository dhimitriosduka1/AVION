import pickle
import argparse

from second_party.storage.sqlite import SQLiteClient
from second_party.preprocess.utils import preprocess_captions


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ego4d-embeddings-path", type=str, required=True)
    parser.add_argument("--lavila-embeddings-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
