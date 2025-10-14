import os
import re
import json
import argparse
import inspect
from tqdm import tqdm
from collections import Counter


def preprocess_captions(captions):
    def lower(text):
        def replacer(match):
            word = match.group()

            # Keep word unchanged if it starts with '#' or is a single character
            if word.startswith("#") or len(word) == 1:
                return word

            return word.lower()

        # Use regex to match words
        return re.sub(r"\b\w+\b", replacer, text)

    results = []

    for c in captions:
        # 1. Strip the caption
        c = c.strip()

        # 2. Replace multiple consecutive spaces with a single space
        c = re.sub(r"\s{2,}", " ", c)

        # 3. Remove punctuation at the end of the line
        c = re.sub(r"[.,!?;:]+$", "", c)

        # 4. Convert to lowercase (only for words, not for hashtags or single characters)
        c = lower(c)

        results.append(c)

    return results


def main(args):
    # First, resolve all the captions path
    captions_paths = []
    for root, dirs, files in tqdm(
        os.walk(args.root_path), desc="Resolving captions paths"
    ):
        for file in files:
            if file == ("captions.json"):
                captions_paths.append(os.path.join(root, file))

    print(f"Found {len(captions_paths)} captions paths")

    # Load all the captions
    captions = []
    for captions_path in tqdm(captions_paths, desc="Loading captions"):
        with open(captions_path, "r") as f:
            data = json.load(f)

            if "metadata" not in data:
                continue

            for m in data["metadata"]:
                captions.extend(preprocess_captions(m["captions"]))

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

    with open(os.path.join(args.root_path, "unique_captions.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
