import argparse
import pickle
import os
import json
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm
from second_party.preprocess.utils import (
    preprocess_caption_v2 as preprocess_captions,
)

def group_samples_by_video(data):
    video_groups = defaultdict(list)
    for idx, sample in enumerate(data):
        video_id, start, end, original_caption = sample
        video_groups[video_id].append((idx, start, end, original_caption))
    return video_groups

def process_one_video(
    video_id,
    gt_segments,
    ego4d_embeddings,
    ego4d_unique_captions,
    lavila_embeddings,
    lavila_unique_captions,
    chunk_metadata_root,
    tau,
):
    """
    Refines temporal boundaries for a single video using text-based semantic similarity.
    """
    seed_embeddings = []
    
    for (gt_idx, start, end, original_caption) in gt_segments:
        processed_caption = preprocess_captions([original_caption])[0]
        idx = ego4d_unique_captions[processed_caption]
        seed_embeddings.append(ego4d_embeddings[idx])
    
    
        
    # 3. Initialize timeline with LaViLa chunks and their embeddings
    timeline = []
    for chunk in all_chunks:
        lavila_cap = chunk["caption"]
        chunk_emb = None
        
        if lavila_cap in lavila_caption_to_idx:
            emb_index = lavila_caption_to_idx[lavila_cap]
            chunk_emb = lavila_embeddings[emb_index]
        
        timeline.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "lavila_caption": lavila_cap,
            "embedding": chunk_emb,
            "label": "NEUTRAL",  # Default label
            "max_similarity": -1.0
        })

    # 4. Seed the timeline: Label chunks that overlap with GT segments
    for gt_idx, gt_start, gt_end, _ in gt_segments:
        for chunk in timeline:
            chunk_start, chunk_end = chunk["start"], chunk["end"]
            
            # Calculate temporal overlap
            overlap = max(0, min(chunk_end, gt_end) - max(chunk_start, gt_start))
            
            # If chunk overlaps, assign it the seed label
            # This implements the constraint: "no segment should expand over another action’s region"
            if overlap > 0:
                chunk["label"] = gt_idx
                
                # Calculate similarity to its own seed for reference
                if gt_idx in seed_embeddings and chunk["embedding"] is not None:
                    sim = np.dot(seed_embeddings[gt_idx], chunk["embedding"])
                    # If it's part of multiple seeds, assign to the most similar
                    if sim > chunk["max_similarity"]:
                        chunk["max_similarity"] = sim
                        chunk["label"] = gt_idx
                # Note: A chunk could overlap two seeds. This logic will assign it
                # to the last one it overlaps with, or the most similar if embeddings are present.
                # For this problem, we assume GT segments are non-overlapping.

    # 5. Identify NEUTRAL gaps
    gaps = []
    current_gap = []
    in_gap = False
    for i, chunk in enumerate(timeline):
        if chunk["label"] == "NEUTRAL":
            if not in_gap:
                in_gap = True
            current_gap.append(i)
        else:
            if in_gap:
                gaps.append(current_gap)
                current_gap = []
                in_gap = False
    if in_gap:
        gaps.append(current_gap)

    # 6. Compete for the gaps (Local Expansion)
    for gap_indices in gaps:
        if not gap_indices:
            continue
            
        first_chunk_idx = gap_indices[0]
        last_chunk_idx = gap_indices[-1]

        # Find nearest *preceding* action label
        preceding_label = None
        if first_chunk_idx > 0:
            preceding_label = timeline[first_chunk_idx - 1]["label"]
        
        # Find nearest *following* action label
        following_label = None
        if last_chunk_idx < len(timeline) - 1:
            following_label = timeline[last_chunk_idx + 1]["label"]

        # Get embeddings for these neighboring actions
        preceding_emb = seed_embeddings.get(preceding_label)
        following_emb = seed_embeddings.get(following_label)

        # 7. Assign chunks in the gap
        for chunk_idx in gap_indices:
            chunk = timeline[chunk_idx]
            if chunk["embedding"] is None:
                continue

            chunk_emb = chunk["embedding"]
            sim_A = -1.0
            sim_B = -1.0

            if preceding_emb is not None:
                sim_A = np.dot(preceding_emb, chunk_emb)
            
            if following_emb is not None:
                sim_B = np.dot(following_emb, chunk_emb)

            # Decision logic: Assign to winner if similarity is above threshold
            if sim_A > sim_B and sim_A >= tau:
                chunk["label"] = preceding_label
                chunk["max_similarity"] = sim_A
            elif sim_B > sim_A and sim_B >= tau:
                chunk["label"] = following_label
                chunk["max_similarity"] = sim_B
            # Else: it remains NEUTRAL

    # 8. Merge consecutive chunks to form new segments
    final_results = []
    if not timeline:
        return []

    current_label = timeline[0]["label"]
    current_start = timeline[0]["start"]
    current_end = timeline[0]["end"]

    for chunk in timeline[1:]:
        if chunk["label"] == current_label:
            current_end = chunk["end"] # Extend the segment
        else:
            # Segment changed, save the previous one if it's not neutral
            if current_label != "NEUTRAL":
                original_caption = gt_idx_to_caption[current_label]
                final_results.append((video_id, current_start, current_end, original_caption))
            
            # Start a new segment
            current_label = chunk["label"]
            current_start = chunk["start"]
            current_end = chunk["end"]

    # Don't forget the last segment
    if current_label != "NEUTRAL":
        original_caption = gt_idx_to_caption[current_label]
        final_results.append((video_id, current_start, current_end, original_caption))

    return final_results
    

def main(args):
    wandb.init(
        project="Thesis",
        name=f"Similarity Based Expansion - tau={args.threshold}",
        config={**args.__dict__},
        group=f"Similarity Based Expansion",
    )

    # Read the dataset
    print("Loading GT dataset...")
    with open(args.dataset, "rb") as f:
        gt_dataset = pickle.load(f)

    # Group samples by video_id
    gt_dataset = group_samples_by_video(gt_dataset)
    print(f"Loaded {len(gt_dataset)} videos.")

    # Load Ego4D Embeddings
    print("Loading Ego4D embeddings...")
    ego4d_embeddings_shape = json.load(
        open(os.path.join(args.dataset_embeddings_path, "shape.json"), "r")
    )
    ego4d_unique_captions = json.load(
        open(os.path.join(args.dataset_embeddings_path, "captions.json"), "r")
    )
    ego4d_embeddings = np.memmap(
        os.path.join(args.dataset_embeddings_path, "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(ego4d_embeddings_shape["shape"]),
    )

    # Load LaViLa Embeddings
    print("Loading LaViLa embeddings...")
    lavila_embeddings_shape = json.load(
        open(os.path.join(args.lavila_embeddings_path, "shape.json"), "r")
    )
    lavila_unique_captions = json.load(
        open(os.path.join(args.lavila_embeddings_path, "captions.json"), "r")
    )
    lavila_embedding = np.memmap(
        os.path.join(args.lavila_embeddings_path, "embeddings.memmap"),
        mode="r",
        dtype=np.float32,
        shape=tuple(lavila_embeddings_shape["shape"]),
    )
    # Create a mapping from caption string to its index in the embedding array
    lavila_caption_to_idx = {cap: i for i, cap in enumerate(lavila_unique_captions)}


    print("Processing videos...")
    results = []
    for video_id, samples in tqdm(gt_dataset.items(), desc="Processing videos"):
        
        video_results = process_one_video(
            video_id=video_id,
            gt_segments=samples,
            ego4d_embeddings=ego4d_embeddings,
            lavila_embeddings=lavila_embedding,
            lavila_caption_to_idx=lavila_caption_to_idx,
            chunk_metadata_root=args.chunk_metadata_root,
            tau=args.threshold,
        )
        results.extend(video_results)

    print(f"Finished processing. Generated {len(results)} new refined segments.")
    
    # Save the results
    output_filename = f"refined_segments_tau{args.threshold}.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Saved refined segments to {output_filename}")
    
    wandb.log({
        "total_new_segments": len(results),
        "total_videos": len(gt_dataset)
    })
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess AVION retrieval results with similarity-based expansion"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Pickle file with samples: (video_id, start, end, caption)",
    )
    parser.add_argument(
        "--dataset-embeddings-path",
        type=str,
        required=True,
        help="Precomputed and normalized Ego4D captions embeddings",
    )
    parser.add_argument(
        "--lavila-embeddings-path",
        type=str,
        required=True,
        help="Precomputed and normalized LaViLa captions embeddings",
    )
    parser.add_argument(
        "--chunk-metadata-root",
        type=str,
        required=True,
        help="Root folder containing per-chunk captions metadata JSONs",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Cosine similarity threshold (tau)"
    )

    args = parser.parse_args()
    main(args)