import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm
import logging
import os
import math

# Suppress HF logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine EgoVLP clips with WandB logging (Multi-GPU)."
    )

    # --- I/O Params ---
    parser.add_argument("--input_path", type=str, default="./data/train_clips.pkl")
    parser.add_argument("--output_path", type=str, default="./data/refined_clips.pkl")

    # --- WandB Params ---
    parser.add_argument("--wandb_project", type=str, default="Thesis")
    parser.add_argument("--run_name", type=str, default="bart-large-h200")

    # --- EgoVLP Logic ---
    parser.add_argument("--min_seconds", type=float, default=0.0)
    parser.add_argument("--max_seconds", type=float, default=10000.0)

    # --- Model Config ---
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--batch_size", type=int, default=256)

    return parser.parse_args()


def setup_distributed():
    """Initializes the distributed process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        # Fallback for single GPU/CPU run
        print("Distributed environment not detected. Running on single device.")
        return 0, 1, 0 if torch.cuda.is_available() else -1


class SemanticScaler:
    def __init__(self, model_name, batch_size, device_id):
        self.batch_size = batch_size
        self.device = device_id

        print(
            f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Loading {model_name} on Device {self.device}..."
        )

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
            framework="pt",
        )
        self.candidate_labels = [
            "short instantaneous action",
            "long continuous process",
        ]

    def compute_lambdas(self, texts):
        lambdas = []
        # Only show progress bar on Rank 0 to avoid clutter
        disable_tqdm = dist.is_initialized() and dist.get_rank() != 0

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc=f"Rank {dist.get_rank() if dist.is_initialized() else 0} Inference",
            disable=disable_tqdm,
        ):
            batch = texts[i : i + self.batch_size]
            results = self.classifier(batch, self.candidate_labels)

            for res in results:
                scores = dict(zip(res["labels"], res["scores"]))
                p_long = scores["long continuous process"]
                p_short = scores["short instantaneous action"]

                # Formula: 2^(P_long - P_short)
                lambdas.append(2.0 ** (p_long - p_short))

        return np.array(lambdas)


def log_distribution_plot(df, args):
    """Generates and logs a comparison plot to WandB (Rank 0 only)."""
    print("Generating distribution plot...")
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    sns.histplot(
        df["duration_base"],
        color="blue",
        label="Original Duration",
        kde=True,
        element="step",
        alpha=0.3,
        bins=50,
        stat="density",
    )
    sns.histplot(
        df["duration_final"],
        color="red",
        label="Refined Duration (Semantic)",
        kde=True,
        element="step",
        alpha=0.3,
        bins=50,
        stat="density",
    )

    plt.title("Clip Duration Distribution (Base=2.0)")
    plt.xlabel("Clip Duration (Seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(0, args.max_seconds + 5)

    wandb.log({"duration_distribution": wandb.Image(plt)})
    plt.close()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    # --- Initialize WandB (Only on Rank 0) ---
    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # --- Load Data ---
    # In this script, all ranks load the pickle independently.
    # Since pickle load is fast, this avoids broadcasting complexity.
    if is_main_process:
        print(f"Loading {args.input_path}...")

    with open(args.input_path, "rb") as f:
        raw_data = pickle.load(f)

    df = pd.DataFrame(raw_data, columns=["video_id", "start", "end", "caption"])
    unique_captions = df["caption"].unique()

    # --- Distribute Work ---
    # Split unique captions roughly equally among ranks
    captions_per_rank = np.array_split(unique_captions, world_size)
    local_captions = captions_per_rank[rank].tolist()

    if is_main_process:
        print(f"Total unique captions: {len(unique_captions)}")
        print(f"Distributing work across {world_size} GPUs.")

    # --- Local Inference ---
    scaler = SemanticScaler(args.model_name, args.batch_size, local_rank)
    local_lambdas = scaler.compute_lambdas(local_captions)

    # Create a local dictionary mapping {caption: lambda}
    local_map = dict(zip(local_captions, local_lambdas))

    # --- Gather Results ---
    # Gather all dictionaries to Rank 0
    if dist.is_initialized():
        gathered_maps = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_maps, local_map)
    else:
        gathered_maps = [local_map]

    # --- Main Process Finalization ---
    if is_main_process:
        # Merge all dictionaries
        full_caption_map = {}
        for m in gathered_maps:
            full_caption_map.update(m)

        print(f"Aggregated results. Total computed: {len(full_caption_map)}")

        # Map lambdas back to main DataFrame
        df["lambda"] = df["caption"].map(full_caption_map)
        wandb.log({"mean_lambda": df["lambda"].mean()})

        # --- Logic for Full Duration Scaling ---
        df["duration_base"] = df["end"] - df["start"]
        df["center"] = (df["start"] + df["end"]) / 2.0
        df["duration_final"] = df["duration_base"] * df["lambda"]
        df["duration_final"] = df["duration_final"].clip(
            lower=args.min_seconds, upper=args.max_seconds
        )
        df["new_start"] = (df["center"] - (df["duration_final"] / 2.0)).clip(lower=0.0)
        df["new_end"] = df["center"] + (df["duration_final"] / 2.0)

        # Logging & Stats
        log_distribution_plot(df, args)
        stats = {
            "avg_duration_original": df["duration_base"].mean(),
            "avg_duration_refined": df["duration_final"].mean(),
            "clips_expanded": (df["lambda"] > 1.05).mean(),
            "clips_shrunk": (df["lambda"] < 0.95).mean(),
        }
        wandb.log(stats)
        print(f"Stats: {stats}")

        # Save Output
        output_data = list(
            df[["video_id", "new_start", "new_end", "caption"]].itertuples(
                index=False, name=None
            )
        )
        print(f"Saving {len(output_data)} clips to {args.output_path}...")
        with open(args.output_path, "wb") as f:
            pickle.dump(output_data, f)

        wandb.finish()
        print("Done.")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
