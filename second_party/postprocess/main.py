# The main file!!!

# NOTE: The embeddings are assumed to be already normalized.

import pickle
import argparse

from second_party.preprocess.utils import preprocess_captions
from second_party.postprocess.safetensor_context import SafeTensorContext


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"
    assert args.ego4d_unique_captions_path.endswith(
        ".safetensors"
    ), "Ego4d unique captions must be a safetensors file"
    assert args.lavila_unique_captions_path.endswith(
        ".safetensors"
    ), "LaViLa unique captions must be a safetensors file"

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    # print(f"Opening {args.ego4d_unique_captions_path} vocabulary")
    # original_ds_vocabulary = SafeTensorContext(
    #     args.ego4d_unique_captions_path, framework="pt", device="cpu"
    # )
    # print(f"Loaded {len(original_ds_vocabulary)} unique captions")

    print(f"Opening {args.lavila_unique_captions_path} vocabulary")
    lavila_vocabulary = SafeTensorContext(
        args.lavila_unique_captions_path, framework="pt", device="cpu"
    )
    print(f"Loaded {len(lavila_vocabulary)} unique captions")

    # print(original_ds_vocabulary.get_tensor(original_ds_vocabulary.keys()[0]))

    # original_ds_vocabulary.close()
    # lavila_vocabulary.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ego4d-unique-captions-path", type=str, required=True)
    parser.add_argument("--lavila-unique-captions-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
