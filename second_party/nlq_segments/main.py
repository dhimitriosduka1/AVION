import argparse
import pickle as pkl
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import decord
from pathlib import Path
import wandb


def plot_distribution(values, title, xlabel="Length (seconds)", ylabel="Frequency"):
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.hist(values, bins=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--output-path", default="/dais/fs/scratch/dduka/databases/ego4d/nlq/", type=str
    )
    parser.add_argument(
        "--video-chunk-root",
        default="/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec",
        type=str,
    )
    return parser.parse_args()


def compute_nlq_segment(start, end, S=5.0, seed=42, duration=9000.0):
    rng = np.random.default_rng(seed)
    Delta = 0.5 * (end - start)
    t_c = 0.5 * (start + end)

    s = rng.uniform(1.0, S)
    T = (s - 1.0) * Delta
    delta_t = rng.uniform(-T, T)

    new_start = max(0.0, (t_c - delta_t) - s * Delta)
    new_end = min(duration, (t_c - delta_t) + s * Delta)

    return new_start, new_end


def probe_duration(path):
    if not path.exists():
        return path, None
    try:
        vr = decord.VideoReader(str(path))
        fps = max(float(vr.get_avg_fps()), 1e-6)
        return path, (len(vr) / fps)
    except Exception:
        return path, None


def main(args):
    wandb.init(
        project="Thesis",
        name=f"NLQ",
        config={**args.__dict__},
        group=f"Refine Dataset NLQ",
    )
    # Load dataset
    with open(args.dataset, "rb") as f:
        data = pkl.load(f)
    print(f"Loaded {len(data)} samples")

    result = []
    old_segments = []
    new_segments = []

    video_id_set = set([sample[0] for sample in data])
    video_lengths = {}
    for video_id in tqdm(video_id_set, desc="Probing video durations..."):
        dir = Path(f"{args.video_chunk_root}/{video_id}.mp4")
        mp4_files = list(dir.rglob("*.mp4"))

        mp4_files.sort()

        base_duration = 15.0 * (len(mp4_files) - 1)
        _, last_duration = probe_duration(mp4_files[-1])

        video_lengths[video_id] = base_duration + last_duration

    for sample in tqdm(data, desc="Computing..."):
        video_id = sample[0]
        start = sample[1]
        end = sample[2]
        caption = sample[3]

        new_start, new_end = compute_nlq_segment(
            start=start, end=end, duration=video_lengths[video_id]
        )
        result.append((video_id, new_start, new_end, caption))

        old_segments.append(end - start)
        new_segments.append(new_end - new_start)

    old_segments = np.array(old_segments)
    new_segments = np.array(new_segments)

    # Write data to file
    with open(f"{args.output_path}ego4d_train_nlq.pkl", "wb") as f:
        pkl.dump(result, f)

    print(f"Mean length (original): {np.mean(old_segments)}")
    print(f"Std length (original): {np.std(old_segments)}")

    print(f"Mean length (after): {np.mean(new_segments)}")
    print(f"Std length (after): {np.std(new_segments)}")

    # Create and save plots
    old_fig = plot_distribution(
        old_segments,
        "Original Distribution",
    )
    old_fig.savefig(f"{args.output_path}original_distribution.png")
    print("Saved 'original_distribution.png'")

    new_fig = plot_distribution(new_segments, "New Distribution")
    new_fig.savefig(f"{args.output_path}new_distribution.png")
    print("Saved 'new_distribution.png'")

    print("\nLogging plots to wandb...")
    wandb.log(
        {"Original Segment Distribution": old_fig, "New Segment Distribution": new_fig}
    )

    plt.close(old_fig)
    plt.close(new_fig)

    print("Saving plots to disk...")
    old_fig.savefig(f"{args.output_path}/original_distribution.png")
    new_fig.savefig(f"{args.output_path}/new_distribution.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)
