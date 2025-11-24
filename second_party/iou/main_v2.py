import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import wandb
import plotly.express as px
import spacy
import concurrent.futures  # --- NEW IMPORT ---

# --- NEW ---
# A global variable to hold the spaCy model for each worker
worker_nlp = None


def init_worker():
    """
    Initializer for each worker process. Loads the spaCy model.
    """
    global worker_nlp
    print("Loading spaCy model in worker process...")
    # Disable components we don't need to speed up loading and processing
    worker_nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])


# --- NEW ---
def process_video_group(video_id, gt_segments, pl_segments, args):
    """
    Processes all segments for a single video.
    This function runs in a separate worker process.
    """
    global worker_nlp  # Access the nlp model loaded by init_worker

    video_results = []
    video_old_segments = []
    video_new_segments = []

    # Iterate through each GROUND TRUTH segment in this video
    for gt_seg in gt_segments:
        original_start = float(gt_seg[1])
        original_end = float(gt_seg[2])
        video_old_segments.append(original_end - original_start)

        best_pl_match = None
        best_hybrid_score = 0.0

        # If there are no PLs for this video, we can skip the search
        if pl_segments:
            # Find the best-matching pseudo-label for this GT segment
            for pl_seg in pl_segments:

                # 1. Check Temporal Overlap (IoU)
                iou = _compute_iou(gt_seg, pl_seg)

                # Optimization: if no/low overlap, don't bother with semantic check
                if iou <= args.min_iou:
                    continue

                temporal_score = iou

                # 2. Check Semantic Score
                # Pass the worker's nlp model to the calculation function
                semantic_score = calculate_semantic_score(
                    gt_seg, pl_seg, worker_nlp, args
                )

                # If semantic check fails, score is 0
                if semantic_score == 0.0:
                    continue

                # 3. Calculate Hybrid Score
                hybrid_score = semantic_score * temporal_score

                if hybrid_score > best_hybrid_score:
                    best_hybrid_score = hybrid_score
                    best_pl_match = pl_seg

        # --- Apply the merge based on the best match ---
        final_start, final_end = original_start, original_end

        if best_pl_match is not None and best_hybrid_score > args.min_hybrid_score:
            final_start = min(original_start, float(best_pl_match[1]))
            final_end = max(original_end, float(best_pl_match[2]))

        video_results.append((video_id, final_start, final_end, gt_seg[3]))
        video_new_segments.append(final_end - final_start)

    # Return the results for this video
    return video_results, video_old_segments, video_new_segments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine GT segments by merging with pseudo-labels based on Hybrid Score (Semantic + IoU)"
    )
    # --- Arguments are unchanged ---
    parser.add_argument(
        "--dataset",
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_enriched.pkl",
        help="Path to ground-truth pickle (e.g., ego4d_train.pkl)",
    )
    parser.add_argument(
        "--pseudolabels",
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_uncovered_all.narrator_63690737.return_5_enriched.pkl",
        help="Path to pseudo-label pickle",
    )
    parser.add_argument(
        "--out-path",
        default="/dais/fs/scratch/dduka/databases/ego4d/hybrid",
        help="Desired output path",
    )
    parser.add_argument(
        "--min-iou",
        default=0.01,
        type=float,
        help="Minimum IoU threshold to even consider a semantic match",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        help="Postfix for wandb logging",
    )
    parser.add_argument(
        "--min-noun-sim",
        default=0.1,
        type=float,
        help="Minimum Jaccard similarity for nouns to pass semantic check",
    )
    parser.add_argument(
        "--min-verb-sim",
        default=0.4,
        type=float,
        help="Minimum spaCy similarity for verbs to pass semantic check",
    )
    parser.add_argument(
        "--min-hybrid-score",
        default=0.15,
        type=float,
        help="Minimum final (Semantic * IoU) score to perform a merge",
    )
    parser.add_argument(
        "--noun-weight",
        default=0.4,
        type=float,
        help="Weight for noun similarity in the semantic score",
    )
    parser.add_argument(
        "--verb-weight",
        default=0.6,
        type=float,
        help="Weight for verb similarity in the semantic score",
    )
    # --- NEW: Add argument for number of workers ---
    parser.add_argument(
        "--num-workers",
        default=None,  # None lets ProcessPoolExecutor decide (usually all cores)
        type=int,
        help="Number of worker processes to use for parallelization",
    )

    return parser.parse_args()


# --- All helper functions below are UNCHANGED ---


def load_data(path):
    print(f"Loading data from {path} ...")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def save_data(path, data):
    print(f"\nSaving refined data to {path} ...")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _group_videos_by_video_id(data):
    video_groups = defaultdict(list)
    for sample in data:
        video_id = sample[0]
        video_groups[video_id].append(sample)
    for video_id in video_groups:
        video_groups[video_id].sort(key=lambda x: float(x[1]))
    return video_groups


def _compute_iou(seg_a, seg_b):
    start_a, end_a = float(seg_a[1]), float(seg_a[2])
    start_b, end_b = float(seg_b[1]), float(seg_b[2])

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(0.0, intersection_end - intersection_start)

    union = (end_a - start_a) + (end_b - start_b) - intersection

    if union == 0:
        return 0.0 if (end_a - start_a) > 0 or (end_b - start_b) > 0 else 1.0

    iou = intersection / union
    return iou


def extract_keywords(caption, nlp):
    """Extracts lemmatized nouns and verbs from a caption."""
    doc = nlp(str(caption).lower())  # Ensure caption is a string
    nouns = []
    verbs = []

    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and token.pos_ in ["NOUN", "PROPN", "VERB"]
        ):
            if token.pos_ in ["NOUN", "PROPN"]:
                nouns.append(token.lemma_)
            elif token.pos_ == "VERB":
                verbs.append(token.lemma_)

    return set(nouns), set(verbs)


def get_jaccard_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0  # Both sets are empty, so they are identical
    return len(intersection) / len(union)


def get_verb_similarity(verbs1_set, verbs2_set, nlp):
    """Calculates the max semantic similarity between two sets of verbs."""
    if not verbs1_set and not verbs2_set:
        return 1.0  # Both are empty, "match"
    if not verbs1_set or not verbs2_set:
        return 0.0  # Only one is empty, "no match"

    max_similarity = 0.0

    for verb1 in verbs1_set:
        doc1 = nlp(verb1)
        if not doc1.has_vector:
            continue
        for verb2 in verbs2_set:
            doc2 = nlp(verb2)
            if not doc2.has_vector:
                continue

            similarity = doc1.similarity(doc2)
            if similarity > max_similarity:
                max_similarity = similarity

    return max_similarity


def calculate_semantic_score(seg_gt, seg_pl, nlp, args):
    """
    Calculates the semantic score based on noun and verb similarity.
    Returns 0.0 if thresholds are not met.
    """
    # Assuming caption is the 4th element (index 3)
    caption_gt = seg_gt[3]
    caption_pl = seg_pl[3]

    nouns_gt, verbs_gt = extract_keywords(caption_gt, nlp)
    nouns_pl, verbs_pl = extract_keywords(caption_pl, nlp)

    noun_sim = get_jaccard_similarity(nouns_gt, nouns_pl)
    verb_sim = get_verb_similarity(verbs_gt, verbs_pl, nlp)

    # --- Enforce minimum thresholds ---
    if noun_sim < args.min_noun_sim:
        return 0.0
    if verb_sim < args.min_verb_sim:
        return 0.0

    # --- Calculate weighted score ---
    score = (args.noun_weight * noun_sim) + (args.verb_weight * verb_sim)
    return score


# --- MODIFIED main() FUNCTION ---
def main(args):
    wandb.init(
        project="Thesis",
        name=(
            f"Hybrid {args.min_hybrid_score} (V:{args.min_verb_sim}, N:{args.min_noun_sim}) - {args.postfix}"
            if args.postfix
            else f"Hybrid {args.min_hybrid_score} (V:{args.min_verb_sim}, N:{args.min_noun_sim})"
        ),
        config={**args.__dict__},
        group=f"Refine Dataset Hybrid",
    )

    # --- spaCy model is no longer loaded in main ---
    # print("Loading spaCy model...")
    # nlp = spacy.load("en_core_web_md")  # Load the model

    ground_truth_data = load_data(args.dataset)
    pseudo_labels_data = load_data(args.pseudolabels)

    if ground_truth_data is None or pseudo_labels_data is None:
        print("Exiting due to file loading error.")
        return

    print(f"Total ground truth segments: {len(ground_truth_data)}")
    print(f"Total pseudo-labels segments: {len(pseudo_labels_data)}")

    print("Grouping videos by video_id...")
    ground_truth_video_groups = _group_videos_by_video_id(ground_truth_data)
    pseudo_labels_video_groups = _group_videos_by_video_id(pseudo_labels_data)

    results = []
    old_segments = []
    new_segments = []

    # --- MODIFIED: Use ProcessPoolExecutor for parallel execution ---
    print(f"Starting parallel processing with {args.num_workers or 'all'} workers...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=init_worker
    ) as executor:

        futures = []
        # Submit all videos as jobs to the pool
        for video_id, gt_segments in ground_truth_video_groups.items():
            pl_segments = pseudo_labels_video_groups.get(video_id, [])
            futures.append(
                executor.submit(
                    process_video_group, video_id, gt_segments, pl_segments, args
                )
            )

        # Collect results as they are completed, with a progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Refining segments",
        ):
            try:
                # Get the results from the completed job
                video_results, video_old, video_new = future.result()

                # Add the results from this video to the main lists
                results.extend(video_results)
                old_segments.extend(video_old)
                new_segments.extend(video_new)
            except Exception as e:
                print(f"An error occurred while processing a video: {e}")

    # --- The rest of the script is UNCHANGED ---

    print(f"Len of original ds: {len(ground_truth_data)}")
    print(f"Len of refined ds: {len(results)}")

    output_filename = f"{args.out_path}/ego4d_train_hybrid_{args.min_hybrid_score}_V{args.min_verb_sim}_N{args.min_noun_sim}.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved refined data to {output_filename}")

    # --- Plotting (unchanged) ---
    fig_old = px.histogram(
        x=old_segments,
        nbins=100,
        title="Original Distribution",
        labels={"x": "Length (seconds)", "y": "Frequency"},
        log_y=True,
    )
    fig_old.update_layout(bargap=0)

    fig_new = px.histogram(
        x=new_segments,
        nbins=100,
        title="New Distribution",
        labels={"x": "Length (seconds)", "y": "Frequency"},
        log_y=True,
    )
    fig_new.update_layout(bargap=0)

    wandb.log(
        {
            "Original Segment Distribution": fig_old,
            "New Segment Distribution": fig_new,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
