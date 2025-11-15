import argparse
import pickle as pkl
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

def plot_distribution(values, title, xlabel = "Length (seconds)", ylabel = "Frequency"):
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
    parser.add_argument("--output-path", default="/dais/fs/scratch/dduka/databases/ego4d/nlq/", type=str)
    return parser.parse_args()

# From https://arxiv.org/pdf/2301.00746
def compute_nlq_segment(start, end, S=5.0, seed=42):
    rng = np.random.default_rng(seed)
    s = rng.uniform(1.0, S)
    tc = (start + end) / 2.0

    delta = (end - start) / 2.0
    T = (s - 1.0) * delta
    delta_t = rng.uniform(-T, T)

    new_start = tc - delta_t - s * delta
    new_end = tc - delta_t + s * delta
    
    return new_start, new_end

def compute_nlq_segment_capped(start, end, S=5.0, seed=42):
    rng = np.random.default_rng(seed)
    Delta = 0.5 * (end - start)
    t_c   = 0.5 * (start + end)

    s = rng.uniform(1.0, S)
    T = (s - 1.0) * Delta
    delta_t = rng.uniform(-T, T)

    return (t_c - delta_t) - s * Delta, (t_c - delta_t) + s * Delta

def main(args):
    # Load dataset
    with open(args.dataset, "rb") as f:
        data = pkl.load(f)
    print(f"Loaded {len(data)} samples")

    result = []
    old_segments = []
    new_segments = []
    for sample in tqdm(data, desc="Computing..."):
        video_id = sample[0]
        start = sample[1]
        end = sample[2]
        caption = sample[3]

        new_start, new_end = compute_nlq_segment(start=start, end=end)
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

    new_fig = plot_distribution(
        new_segments,
        "New Distribution"
    )
    new_fig.savefig(f"{args.output_path}new_distribution.png")
    print("Saved 'new_distribution.png'")

if __name__ == "__main__":
    args = parse_args()
    main(args)
