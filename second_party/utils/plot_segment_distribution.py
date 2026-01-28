import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")


def convert_to_supported_format(path):
    import pandas as pd

    df = pd.read_csv(path)

    data = []
    for row in df.iterrows():
        row_data = row[1]
        data.append(
            (
                row_data["uuid"],
                row_data["video_id"],
                row_data["start_s"],
                row_data["end_s"],
                row_data["caption"],
            )
        )

    return data


def get_segment_lengths(data):
    row_elements = len(data[0])
    start_idx = 1 if row_elements == 4 else 2
    end_idx = 2 if row_elements == 4 else 3

    segment_lengths = []
    for sample in data:
        segment_lengths.append(sample[end_idx] - sample[start_idx])

    return segment_lengths


def plot_segment_distribution(
    segment_lengths, output_path=None, log_scale=False, title=""
):
    """Plot the distribution of segment lengths as a histogram."""
    plt.figure(figsize=(10, 6))

    plt.hist(segment_lengths, bins=50, edgecolor="black", alpha=0.7)

    plt.xlabel("Segment Length", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    title = f"Distribution of Segment Lengths: {title}"
    if log_scale:
        plt.yscale("log")
        title += " (Log Scale)"
    plt.title(title, fontsize=14)

    # Add statistics
    mean_len = np.mean(segment_lengths)
    median_len = np.median(segment_lengths)
    std_len = np.std(segment_lengths)

    stats_text = f"Mean: {mean_len:.2f}\nMedian: {median_len:.2f}\nStd: {std_len:.2f}\nCount: {len(segment_lengths)}"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the distribution of segment lengths from a pickle file."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the pickle file containing segment data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/u/dduka/project/AVION/images",
        help="Path to save the output plot (optional, defaults to <input_filename>.png)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use log scale for the y-axis",
    )

    args = parser.parse_args()

    base_name = args.path.split("/")[-1].split(".")[0]
    args.output = f"{args.output}/{base_name}.png"

    if ".csv" in args.path:
        data = convert_to_supported_format(args.path)
    else:
        with open(args.path, "rb") as f:
            data = pkl.load(f)

    segment_lengths = get_segment_lengths(data)
    print(f"Total segments: {len(segment_lengths)}")
    print(f"Min length: {min(segment_lengths):.2f}")
    print(f"Max length: {max(segment_lengths):.2f}")

    plot_segment_distribution(segment_lengths, args.output, args.log, title=base_name)


if __name__ == "__main__":
    main()
