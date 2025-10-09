import argparse
import pickle
import numpy as np


def extract_timestamps(data):
    return [
        {
            "t_start": sample[1],
            "t_end": sample[2],
        }
        for sample in data
    ]


def segment_length(timestamps):
    return [timestamp["t_end"] - timestamp["t_start"] for timestamp in timestamps]


def main(args):
    dataset_path = args.dataset_path

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    timestamps = extract_timestamps(data)

    segment_lengths = segment_length(timestamps)

    statistics = {
        "mean": np.mean(segment_lengths),
        "median": np.median(segment_lengths),
        "min": np.min(segment_lengths),
        "max": np.max(segment_lengths),
        "std": np.std(segment_lengths),
    }

    print(statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
