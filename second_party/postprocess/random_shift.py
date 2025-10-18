import pickle
import os
from pathlib import Path
import random
import argparse
import numpy as np

from tqdm import tqdm

# Set the random seed
random.seed(42)


def jitter_scale_window(
    start: float,
    end: float,
    scale_min: float = 0.6,
    scale_max: float = 1.6,
    min_duration: float = 1.0,
    max_duration: float = 5.0,
):
    """
    Keep center fixed. Grow/shrink window by a random scale s ~ U[scale_min, scale_max].
    Enforce duration bounds: min_duration <= new_duration <= max_duration.
    """
    assert (
        scale_min > 0 and scale_max > 0 and scale_min <= scale_max
    ), "Invalid scale range"
    c = 0.5 * (start + end)
    d = max(end - start, 1e-6)  # avoid zero

    s = random.uniform(scale_min, scale_max)
    new_d = max(min(d * s, max_duration), min_duration)

    new_start = c - new_d / 2.0
    new_end = c + new_d / 2.0
    return new_start, new_end


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    old_timestamps_duration = []
    new_timestamps_duration = []

    new_data = []
    for sample in tqdm(data, desc="Jittering timestamps"):
        new_start, new_end = jitter_scale_window(
            sample[1],
            sample[2],
            scale_min=0.1,
            scale_max=20.0,
            min_duration=1.0,
            max_duration=5.0,
        )
        old_timestamps_duration.append(sample[2] - sample[1])
        new_timestamps_duration.append(new_end - new_start)
        new_data.append((sample[0], new_start, new_end, sample[3]))

    print(f"Old timestamps duration: {np.mean(old_timestamps_duration)}")
    print(f"Old timestamps duration: {np.std(old_timestamps_duration)}")

    print(f"New timestamps duration: {np.mean(new_timestamps_duration)}")
    print(f"New timestamps duration: {np.std(new_timestamps_duration)}")

    with open(
        os.path.join(
            Path(args.output_path) / "random_shift_timestamps", "ego4d_train_random_shift.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(new_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
