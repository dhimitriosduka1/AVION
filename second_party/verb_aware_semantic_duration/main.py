import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os
import gc
import torch.nn.functional as F

# Suppress HF logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine EgoVLP clips via Conservative Likert Scaling."
    )
    parser.add_argument("--input_path", type=str, default="./data/train_clips.pkl")
    parser.add_argument("--output_path", type=str, default="./data/refined_clips.pkl")

    parser.add_argument(
        "--wandb_project", type=str, default="Verb Aware Semantic Duration"
    )
    parser.add_argument(
        "--run_name", type=str, default="qwen-7b-conservative-overlap-check"
    )

    # Logic
    parser.add_argument("--min_seconds", type=float, default=0.5)
    parser.add_argument("--max_seconds", type=float, default=1000.0)

    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0 if torch.cuda.is_available() else -1


def get_likert_token_ids(tokenizer):
    """
    Get token IDs for '1', '2', '3', '4', '5'.
    """
    options = ["1", "2", "3", "4", "5"]
    token_map = {}
    for t in options:
        ids = tokenizer.encode(t, add_special_tokens=False)
        token_map[t] = ids[-1]

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Likert Token IDs: {token_map}")
    return token_map


class LikertPromptDataset(Dataset):
    def __init__(self, df, tokenizer, system_prompt):
        self.data = df.to_dict("records")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        curr_caption = row["caption"] if row["caption"] else "Unknown Action"

        user_content = (
            f"Task: Rate the temporal nature of the action on a scale from 1 to 5.\n\n"
            f'Action: "{curr_caption}"\n\n'
            f"Scale Definitions:\n"
            f"1. Instantaneous (Impulse, e.g., 'hit', 'drop', 'touch')\n"
            f"2. Very Short (Brief moment, e.g., 'glance', 'grab')\n"
            f"3. Short (Standard action, e.g., 'open door', 'stand up')\n"
            f"4. Medium (Process, e.g., 'mixing', 'cutting')\n"
            f"5. Long (Continuous state, e.g., 'cooking', 'walking', 'reading')\n\n"
            f"Return only the number (1-5)."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        tokens = self.tokenizer(
            full_prompt, truncation=True, max_length=1024, return_tensors="pt"
        )
        return tokens.input_ids.squeeze(0)


class LeftPadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(x.size(0) for x in batch)
        padded_batch = []
        attention_masks = []

        for x in batch:
            seq_len = x.size(0)
            pad_len = max_len - seq_len
            if pad_len > 0:
                pads = torch.full((pad_len,), self.pad_token_id, dtype=x.dtype)
                padded_x = torch.cat([pads, x])
                mask = torch.cat([torch.zeros(pad_len), torch.ones(seq_len)])
            else:
                padded_x = x
                mask = torch.ones(seq_len)
            padded_batch.append(padded_x)
            attention_masks.append(mask)

        input_ids = torch.stack(padded_batch)
        attention_mask = torch.stack(attention_masks).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        print(f"Loading {args.input_path}...")

    # --- Load Data ---
    with open(args.input_path, "rb") as f:
        raw_data = pickle.load(f)

    df = pd.DataFrame(raw_data, columns=["video_id", "start", "end", "caption"])

    # Sort crucial for overlaps
    df = df.sort_values(by=["video_id", "start"])

    indices = np.arange(len(df))
    my_indices = np.array_split(indices, world_size)[rank]
    local_df = df.iloc[my_indices].copy()
    del df
    gc.collect()

    print(f"[Rank {rank}] Processing {len(local_df)} samples.")

    # --- Model ---
    device = f"cuda:{local_rank}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    likert_ids_map = get_likert_token_ids(tokenizer)
    target_ids = [likert_ids_map[k] for k in ["1", "2", "3", "4", "5"]]
    target_tensor = torch.tensor(target_ids, device=device)

    system_prompt = "You are an expert semantic analyst."

    dataset = LikertPromptDataset(local_df, tokenizer, system_prompt)
    collator = LeftPadCollator(pad_token_id=tokenizer.pad_token_id)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    ).eval()

    # --- Inference ---
    local_lambdas = []
    local_scores = []

    disable_tqdm = rank != 0
    debug_printed = False

    for batch in tqdm(dataloader, desc=f"Rank {rank} Inference", disable=disable_tqdm):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

            if not debug_printed and is_main_process:
                print("\n" + "=" * 30)
                print("DEBUG: Checking Likert Output")
                print("=" * 30)
                topk_vals, topk_indices = torch.topk(next_token_logits[0], 5)
                print(
                    f"Top 5 Preds: {[(tokenizer.decode([idx]), round(score.item(), 2)) for score, idx in zip(topk_vals, topk_indices)]}"
                )
                debug_printed = True

            relevant_logits = next_token_logits[:, target_tensor]
            probs = F.softmax(relevant_logits, dim=1)

            weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
            expected_score = torch.sum(probs * weights, dim=1)

            scores_np = expected_score.float().cpu().numpy()
            local_scores.extend(scores_np)

            xp = [1.0, 3.0, 5.0]
            fp = [1.0, 1.5, 2.2]
            batch_lambdas = np.interp(scores_np, xp, fp)
            local_lambdas.extend(batch_lambdas)

    local_lambdas = np.array(local_lambdas)
    local_scores = np.array(local_scores)

    # --- Gather & Save ---
    if dist.is_initialized():
        gathered_lambdas = [None for _ in range(world_size)]
        gathered_scores = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_lambdas, local_lambdas)
        dist.all_gather_object(gathered_scores, local_scores)

        if is_main_process:
            full_lambdas = np.concatenate(gathered_lambdas)
            full_scores = np.concatenate(gathered_scores)
    else:
        full_lambdas = local_lambdas
        full_scores = local_scores

    if is_main_process:
        print("Processing final durations and overlaps...")
        with open(args.input_path, "rb") as f:
            raw_data = pickle.load(f)

        df_final = pd.DataFrame(
            raw_data, columns=["video_id", "start", "end", "caption"]
        )

        if len(full_lambdas) != len(df_final):
            df_final = df_final.iloc[: len(full_lambdas)].copy()

        df_final["lambda"] = full_lambdas
        df_final["likert_score"] = full_scores

        # 1. Calculate Refined Durations
        df_final["duration_base"] = df_final["end"] - df_final["start"]
        df_final["target_duration"] = (
            df_final["duration_base"] * df_final["lambda"]
        ).clip(args.min_seconds, args.max_seconds)

        center = (df_final["start"] + df_final["end"]) / 2.0
        half = df_final["target_duration"] / 2.0

        df_final["new_start"] = center - half
        df_final["new_end"] = center + half
        df_final["duration_final"] = df_final["new_end"] - df_final["new_start"]

        # 2. Compute Overlaps (Original vs Refined)
        df_final = df_final.sort_values(by=["video_id", "start"])

        # A. Original Overlap
        # prev_end is the end of the previous clip in original data
        df_final["prev_end_orig"] = (
            df_final.groupby("video_id")["end"].shift(1).fillna(0.0)
        )
        df_final["overlap_seconds_orig"] = (
            df_final["prev_end_orig"] - df_final["start"]
        ).clip(lower=0.0)

        # B. Refined Overlap
        # Sort by new_start to handle potential re-ordering if expansions are massive (unlikely but safe)
        df_final = df_final.sort_values(by=["video_id", "new_start"])
        df_final["prev_end_refined"] = (
            df_final.groupby("video_id")["new_end"].shift(1).fillna(0.0)
        )
        df_final["overlap_seconds_refined"] = (
            df_final["prev_end_refined"] - df_final["new_start"]
        ).clip(lower=0.0)

        # 3. Create Comparison Bar Plot
        mean_overlap_orig = df_final["overlap_seconds_orig"].mean()
        mean_overlap_refined = df_final["overlap_seconds_refined"].mean()

        # Create a tiny dataframe specifically for this plot
        plot_data = [["Original", mean_overlap_orig], ["Refined", mean_overlap_refined]]
        table_overlap = wandb.Table(
            data=plot_data, columns=["Version", "Mean Overlap (s)"]
        )

        # Use wandb.plot.bar for a clean comparison
        bar_plot = wandb.plot.bar(
            table_overlap,
            "Version",
            "Mean Overlap (s)",
            title="Mean Overlap Comparison (Before vs After)",
        )

        # Log scalars and the single plot
        wandb.log(
            {
                "mean_lambda": df_final["lambda"].mean(),
                "mean_score": df_final["likert_score"].mean(),
                "mean_overlap_orig": mean_overlap_orig,
                "mean_overlap_refined": mean_overlap_refined,
                "charts/overlap_comparison": bar_plot,
            }
        )

        output_data = list(
            df_final[["video_id", "new_start", "new_end", "caption"]].itertuples(
                index=False, name=None
            )
        )
        print(f"Saving {len(output_data)} clips to {args.output_path}...")
        with open(args.output_path, "wb") as f:
            pickle.dump(output_data, f)

        print("Done.")
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
