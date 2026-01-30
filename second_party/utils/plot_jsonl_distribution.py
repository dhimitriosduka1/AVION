import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")

def main(file_path, output_path):
    segment_lengths = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            if "model_output" not in data:
                continue

            model_output = data["model_output"]

            if "start" not in model_output or "end" not in model_output:
                continue

            start = model_output["start"]
            end = model_output["end"]

            if start > end:
                continue

            segment_lengths.append(end - start)

    print(f"Total segments: {len(segment_lengths)}")

    # Get the name of the file from the file_path
    file_name = file_path.split("/")[-1].split(".")[0]
    output_path = f"{output_path}/{file_name}_segment_length_distribution.png"

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(segment_lengths, bins=100)
    plt.title(f"Distribution of Segment Lengths for {file_name}")
    plt.xlabel("Segment Length")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig(output_path)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot distributions from a JSONL file")
    parser.add_argument(
        "--file_path", "-f", required=True, help="Path to the JSONL file"
    )
    parser.add_argument(
        "--output_path", "-o", required=True, help="Path to save the output plots"
    )
    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path

    main(file_path, output_path)
