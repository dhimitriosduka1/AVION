import argparse
import pandas as pd
import numpy as np
import difflib
import torch
import wandb
import os
import glob
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# --- 1. Qwen Summarizer Class ---
class QwenSummarizer:
    def __init__(self, model_id, batch_size=8, local_rank=0):
        # We only print loading info on the main process
        if local_rank == 0:
            print(f"Loading model: {model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.batch_size = batch_size

        # FORCE model to specific GPU based on local_rank
        # This prevents the model from trying to spread across all GPUs
        device = torch.device(f"cuda:{local_rank}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # Critical: Map to specific device, not "auto"
            device_map={"": device},
            attn_implementation="flash_attention_2",
        )

    def summarize_batch(self, text_pairs, rank):
        """
        Generates summaries.
        """
        results = []
        system_prompt = (
            "You are an expert video logger. "
            "Merge the following two sequential egocentric video captions into a single, concise event description. "
            "Combine the actions fluently. Do not include timestamps. Output ONLY the summary."
        )

        # Only show progress bar on Rank 0 to avoid console spam
        iterator = range(0, len(text_pairs), self.batch_size)
        if rank == 0:
            iterator = tqdm(
                iterator,
                desc=f"Rank {rank} Processing",
                total=(len(text_pairs) + self.batch_size - 1) // self.batch_size,
            )

        for i in iterator:
            current_pairs = text_pairs[i : i + self.batch_size]

            batch_prompts = []
            for t1, t2 in current_pairs:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Caption 1: {t1}\nCaption 2: {t2}"},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_prompts.append(text)

            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=64, temperature=0.7, do_sample=True
                )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            results.extend(batch_responses)

            # WandB Log (Only Rank 0 logs progress to keep it clean)
            if rank == 0:
                wandb.log({"progress_percent": (i / len(text_pairs)) * 100})

        return results


# --- 2. Main Processing Logic ---
def process_data(args):
    # --- DDP SETUP ---
    # torchrun sets these environment variables automatically
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    # Initialize WandB only on Rank 0
    if rank == 0:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        print(f"Loading data from {args.input_path}...")

    # Load Data (Every process loads the DF, but processing is cheap)
    try:
        df = pd.read_pickle(args.input_path)
    except Exception as e:
        if rank == 0:
            print(f"Error loading pickle: {e}")
        return

    if isinstance(df, list):
        df = pd.DataFrame(
            df, columns=["video_uuid", "start_time", "end_time", "caption"]
        )

    if "is_augmented" not in df.columns:
        df["is_augmented"] = False

    # --- IDENTIFY CANDIDATES (Replicated on all ranks) ---
    # We do this on all ranks so they all have the same list to split from
    if rank == 0:
        print("Grouping by UUID and identifying merge pairs...")

    candidates = []

    # Sort and Group logic
    grouped = df.groupby("video_uuid")

    # We convert groupby to list to ensure deterministic order across processes
    # (Groupby iteration order is usually stable, but explicit is safer)
    all_groups = sorted(list(grouped), key=lambda x: x[0])

    for video_id, group in all_groups:
        group = group.sort_values("start_time").reset_index(drop=True)
        for i in range(0, len(group) - 1, 2):
            curr_row = group.iloc[i]
            next_row = group.iloc[i + 1]
            candidates.append(
                {
                    "s1": str(curr_row["caption"]),
                    "s2": str(next_row["caption"]),
                    "video_uuid": video_id,
                    "new_start": curr_row["start_time"],
                    "new_end": next_row["end_time"],
                }
            )

    if rank == 0:
        print(f"Total candidates found: {len(candidates)}")
        wandb.log({"total_candidates": len(candidates)})

    if len(candidates) == 0:
        return

    # --- SHARD DATA (The Magic Part) ---
    # Each rank takes every Nth item (0, 4, 8... / 1, 5, 9... etc)
    my_candidates = candidates[rank::world_size]

    print(f"[Rank {rank}] Processing {len(my_candidates)} items...")

    # Initialize Model (Local Rank)
    summarizer = QwenSummarizer(
        args.model, batch_size=args.batch_size, local_rank=local_rank
    )

    # Run Summarization
    text_pairs = [(c["s1"], c["s2"]) for c in my_candidates]
    summaries = summarizer.summarize_batch(text_pairs, rank)

    # Build Partial DataFrame
    new_rows = []
    for candidate, summary in zip(my_candidates, summaries):
        new_rows.append(
            {
                "video_uuid": candidate["video_uuid"],
                "start_time": candidate["new_start"],
                "end_time": candidate["new_end"],
                "caption": summary.strip(),
                "is_augmented": True,
            }
        )

    partial_df = pd.DataFrame(new_rows)

    # --- SAVE PARTIAL RESULTS ---
    # Ensure output dir exists (safely)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_filename = f"{args.output_path}.part_{rank}"
    partial_df.to_pickle(temp_filename)
    print(f"[Rank {rank}] Saved partial to {temp_filename}")

    # Wait for all processes to finish writing files
    dist.barrier()

    # --- MERGE ON RANK 0 ---
    if rank == 0:
        print("Merging all partial files...")
        all_new_rows = []

        # Load all parts
        for r in range(world_size):
            fname = f"{args.output_path}.part_{r}"
            if os.path.exists(fname):
                part_df = pd.read_pickle(fname)
                all_new_rows.append(part_df)
                os.remove(fname)  # Clean up

        if all_new_rows:
            augmented_df = pd.concat(all_new_rows, ignore_index=True)
            final_df = pd.concat([df, augmented_df], ignore_index=True)

            # Logging
            stats = {
                "original_count": len(df),
                "new_samples_added": len(augmented_df),
                "augmentation_ratio": len(augmented_df) / len(df),
            }
            print(f"Stats: {stats}")
            wandb.log(stats)

            final_name = f"{args.output_path.replace('.pkl', '')}_{args.model.split('/')[-1]}.pkl"
            final_df.to_pickle(final_name)
            print(f"Final dataset saved to {final_name}")

            # Log Table Example (Just top 100)
            result_table = wandb.Table(
                columns=["Video UUID", "Input 1", "Input 2", "Merged Summary"]
            )
            # (Simplification: Just logging first few from the augmented set)
            for _, row in augmented_df.head(50).iterrows():
                result_table.add_data(row["video_uuid"], "n/a", "n/a", row["caption"])
            wandb.log({"merge_examples": result_table})

        wandb.finish()

    # Clean exit
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="ego4d_pairwise.pkl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--wandb-project", type=str, default="Thesis")
    parser.add_argument("--wandb-run-name", type=str, default="qwen-ddp-merge")

    args = parser.parse_args()

    process_data(args)
