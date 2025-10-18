import pickle
import os
from pathlib import Path
import random
import argparse
import numpy as np
import decord
import wandb

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
    min_start: float = 0.0,
    video_duration: float = 1.0,
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

    new_start = max(c - new_d / 2.0, min_start)
    new_end = c + new_d / 2.0

    if new_end > video_duration:
        print(f"New end is greater than video duration: {new_end} > {video_duration}")

        new_start += new_end - video_duration
        new_end = video_duration

    return new_start, new_end


def main(args):
    assert args.dataset.endswith(".pkl"), "Dataset must be a pickle file"

    wandb.init(
        project="Thesis",
        name=f"Random Shift - {args.scale_min} - {args.scale_max} - {args.min_duration} - {args.max_duration}",
        config={**args.__dict__},
        group=f"Random Timestamp Shift",
    )

    print(f"Opening {args.dataset} dataset")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    old_timestamps_duration = []
    new_timestamps_duration = []

    new_data = []
    prev_video_id = None
    for i, sample in tqdm(enumerate(data), desc="Jittering timestamps"):
        if prev_video_id != sample[0]:
            vr = decord.VideoReader(os.path.join(args.video_root, f"{sample[0]}.mp4"))
            video_duration = len(vr) / vr.get_avg_fps()
            prev_video_id = sample[0]

        new_start, new_end = jitter_scale_window(
            sample[1],
            sample[2],
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            video_duration=video_duration,
        )
        old_timestamps_duration.append(sample[2] - sample[1])
        new_timestamps_duration.append(new_end - new_start)
        new_data.append((sample[0], new_start, new_end, sample[3]))

        wandb.log(
            {
                "progress": i / len(data),
            }
        )
    with open(
        os.path.join(
            Path(args.output_path) / "random_shift_timestamps",
            "ego4d_train_random_shift.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(new_data, f)

    table = wandb.Table(columns=["Metric", "Mean", "Std"])

    table.add_data(
        "Timestamp Duration (Original)",
        np.mean(old_timestamps_duration),
        np.std(old_timestamps_duration),
    )

    table.add_data(
        "Timestamp Duration (Jittered)",
        np.mean(new_timestamps_duration),
        np.std(new_timestamps_duration),
    )

    wandb.log({"table": table})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=20.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    args = parser.parse_args()

    main(args)
