import os
import json
import argparse
import inspect
import pickle
from tqdm import tqdm
from collections import Counter
from second_party.preprocess.utils import preprocess_captions, preprocess_caption_v2


def main(args):
    if args.preprocess_function == "preprocess_captions":
        from second_party.preprocess.utils import preprocess_captions
    elif args.preprocess_function == "preprocess_caption_v2":
        from second_party.preprocess.utils import (
            preprocess_caption_v2 as preprocess_captions,
        )
    else:
        raise ValueError(f"Invalid preprocess function: {args.preprocess_function}")

    # First, resolve all the captions path
    with open(args.root_path, "rb") as f:
        data = pickle.load(f)

    captions = []
    for sample in tqdm(data, desc="Processing captions"):
        captions.extend(preprocess_captions([sample[-1]]))

    print(f"Loaded {len(captions)} captions")

    captions_counter = Counter(captions)

    # Resolve all the unique captions
    unique_captions = list(set(captions))

    results = {
        "number_of_total_captions": len(captions),
        "number_of_unique_captions": len(unique_captions),
        "percentage_of_unique_captions": len(unique_captions) / len(captions),
        "unique_captions": [],
        "preprocess_function": {
            "source": inspect.getsource(preprocess_captions).strip(),
        },
    }

    for caption, count in captions_counter.items():
        results["unique_captions"].append(
            {
                "text": caption,
                "frequency": count,
            }
        )

    with open(os.path.join(args.output_path, f"unique_captions_{args.preprocess_function}.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--preprocess-function", type=str, default="preprocess_captions"
    )
    args = parser.parse_args()
    main(args)
