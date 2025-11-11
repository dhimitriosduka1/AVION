import math
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import wandb

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine GT segments by optimally assigning in-between pseudo-labels without crossing GTs."
    )
    parser.add_argument(
        "--gt",
        default="/BS/dduka/work/projects/AVION/ego4d_train_enriched.pkl",
        help="Path to ground-truth pickle (e.g., ego4d_train.pkl)",
    )
    parser.add_argument(
        "--pseudolabels",
        default="/BS/dduka/work/projects/AVION/ego4d_train_uncovered_all.narrator_63690737.return_5_enriched_and_filtered.pkl",
        help="Path to pseudo-label pickle",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Desired output path",
    )
    parser.add_argument(
        "--gap",
        default=1.5,
        type=float,
        help="Max allowable gap (seconds) when chaining adjacent segments",
    )
    parser.add_argument(
        "--use-only",
        default=None,
        choices=["nouns", "verbs", None],
        help="Whether to use only nouns or verbs for the overlap check",
    )
    # scoring weights (can be left as defaults)
    parser.add_argument(
        "--alpha_nouns", type=float, default=1.0, help="Weight for noun Jaccard"
    )
    parser.add_argument(
        "--beta_verbs", type=float, default=1.0, help="Weight for verb Jaccard"
    )
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.0,
        help="Minimum score to assign to A/B (else goes to Ø)",
    )
    return parser.parse_args()


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


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _to_sets(sample):
    """Return (nouns, verbs, noun_lemmas, verb_lemmas) as sets (may be empty)."""
    nouns = set(_flatten(sample[4])) if len(sample) > 4 else set()
    verbs = set(_flatten(sample[5])) if len(sample) > 5 else set()
    noun_lemmas = set(_flatten(sample[6])) if len(sample) > 6 else set()
    verb_lemmas = set(_flatten(sample[7])) if len(sample) > 7 else set()
    return nouns, verbs, noun_lemmas, verb_lemmas


def _has_overlap(
    gt_nouns,
    gt_verbs,
    gt_noun_lemmas,
    gt_verb_lemmas,
    cand_nouns,
    cand_verbs,
    cand_noun_lemmas,
    cand_verb_lemmas,
    use_only=None,
):
    # Same logic as your original, but works on precomputed sets
    if not gt_nouns or not cand_nouns:
        nouns_to_check_gt = gt_noun_lemmas
        nouns_to_check_cand = cand_noun_lemmas
    else:
        nouns_to_check_gt = gt_nouns
        nouns_to_check_cand = cand_nouns

    if not gt_verbs or not cand_verbs:
        verbs_to_check_gt = gt_verb_lemmas
        verbs_to_check_cand = cand_verb_lemmas
    else:
        verbs_to_check_gt = gt_verbs
        verbs_to_check_cand = cand_verbs

    if use_only is None:
        has_noun_overlap = bool(nouns_to_check_gt & nouns_to_check_cand)
        has_verb_overlap = bool(verbs_to_check_gt & verbs_to_check_cand)
    elif use_only == "nouns":
        has_noun_overlap = bool(nouns_to_check_gt & nouns_to_check_cand)
        has_verb_overlap = True
    elif use_only == "verbs":
        has_noun_overlap = True
        has_verb_overlap = bool(verbs_to_check_gt & verbs_to_check_cand)
    else:
        raise ValueError(f"Invalid use_only: {use_only}")

    return has_noun_overlap and has_verb_overlap


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union > 0 else 0.0


def _build_tagged_timeline(gt_groups, pl_groups):
    """
    Build merged per-video timelines with explicit source tags.
    Each element is a dict:
      {video_id, start, end, caption, nouns, verbs, noun_lemmas, verb_lemmas, src}
    """
    merged = defaultdict(list)
    vids = set(gt_groups.keys()) | set(pl_groups.keys())
    for vid in vids:
        for s in gt_groups.get(vid, []):
            nouns, verbs, nlem, vlem = _to_sets(s)
            merged[vid].append(
                dict(
                    video_id=vid,
                    start=float(s[1]),
                    end=float(s[2]),
                    caption=s[3] if len(s) > 3 else None,
                    nouns=nouns,
                    verbs=verbs,
                    noun_lemmas=nlem,
                    verb_lemmas=vlem,
                    src="gt",
                )
            )
        for s in pl_groups.get(vid, []):
            nouns, verbs, nlem, vlem = _to_sets(s)
            merged[vid].append(
                dict(
                    video_id=vid,
                    start=float(s[1]),
                    end=float(s[2]),
                    caption=s[3] if len(s) > 3 else None,
                    nouns=nouns,
                    verbs=verbs,
                    noun_lemmas=nlem,
                    verb_lemmas=vlem,
                    src="pl",
                )
            )
        merged[vid].sort(key=lambda x: x["start"])
    return merged


def _compute_scores_and_masks(pls, A, B, gap, use_only, alpha, beta):
    """
    For PLs between A and B (sorted), compute:
      - sA[i], sB[i], s0[i]=0
      - feasibility masks via left/right chainability with 'gap' and overlap.
    """
    n = len(pls)
    sA = [-float("inf")] * n
    sB = [-float("inf")] * n

    def feasible_to_A(i):
        return _has_overlap(
            A["nouns"],
            A["verbs"],
            A["noun_lemmas"],
            A["verb_lemmas"],
            pls[i]["nouns"],
            pls[i]["verbs"],
            pls[i]["noun_lemmas"],
            pls[i]["verb_lemmas"],
            use_only=use_only,
        )

    def feasible_to_B(i):
        return _has_overlap(
            B["nouns"],
            B["verbs"],
            B["noun_lemmas"],
            B["verb_lemmas"],
            pls[i]["nouns"],
            pls[i]["verbs"],
            pls[i]["noun_lemmas"],
            pls[i]["verb_lemmas"],
            use_only=use_only,
        )

    def score_to_A(i):
        An = A["nouns"] if A["nouns"] and pls[i]["nouns"] else A["noun_lemmas"]
        Av = A["verbs"] if A["verbs"] and pls[i]["verbs"] else A["verb_lemmas"]
        Pn = (
            pls[i]["nouns"] if A["nouns"] and pls[i]["nouns"] else pls[i]["noun_lemmas"]
        )
        Pv = (
            pls[i]["verbs"] if A["verbs"] and pls[i]["verbs"] else pls[i]["verb_lemmas"]
        )
        return alpha * _jaccard(An, Pn) + beta * _jaccard(Av, Pv)

    def score_to_B(i):
        Bn = B["nouns"] if B["nouns"] and pls[i]["nouns"] else B["noun_lemmas"]
        Bv = B["verbs"] if B["verbs"] and pls[i]["verbs"] else B["verb_lemmas"]
        Pn = (
            pls[i]["nouns"] if B["nouns"] and pls[i]["nouns"] else pls[i]["noun_lemmas"]
        )
        Pv = (
            pls[i]["verbs"] if B["verbs"] and pls[i]["verbs"] else pls[i]["verb_lemmas"]
        )
        return alpha * _jaccard(Bn, Pn) + beta * _jaccard(Bv, Pv)

    # Left side (A): can we grow from A.end through contiguous PLs to index i?
    left_reach = [False] * n
    for i in range(n):
        if not feasible_to_A(i):
            left_reach[i] = False
            continue
        if i == 0:
            left_reach[i] = abs(pls[i]["start"] - A["end"]) <= gap
        else:
            left_reach[i] = left_reach[i - 1] and (
                abs(pls[i]["start"] - pls[i - 1]["end"]) <= gap
            )
        if left_reach[i]:
            sA[i] = score_to_A(i)

    # Right side (B): can we grow from B.start backwards through contiguous PLs to index i?
    right_reach = [False] * n
    for i in reversed(range(n)):
        if not feasible_to_B(i):
            right_reach[i] = False
            continue
        if i == n - 1:
            right_reach[i] = abs(B["start"] - pls[i]["end"]) <= gap
        else:
            right_reach[i] = right_reach[i + 1] and (
                abs(pls[i + 1]["start"] - pls[i]["end"]) <= gap
            )
        if right_reach[i]:
            sB[i] = score_to_B(i)

    s0 = [0.0] * n
    return sA, s0, sB


def _dp_assign_between(pls, sA, s0, sB, tau_min):
    """
    Monotone DP with states A, Ø, B and transitions:
       A -> A or Ø
       Ø -> Ø or B
       B -> B
    Returns labels in {"A", "0", "B"} of length n.
    Robust against unreachable states (no None backpointers).
    """
    n = len(pls)
    if n == 0:
        return []

    NEG_INF = -1e18

    # Apply threshold: if below tau_min, forbid assignment to that side.
    sA = [x if (x is not None and x >= tau_min) else NEG_INF for x in sA]
    sB = [x if (x is not None and x >= tau_min) else NEG_INF for x in sB]
    s0 = [0.0 if (x is None) else float(x) for x in s0]  # Ø stays 0 by design

    dpA = [NEG_INF] * n
    dp0 = [NEG_INF] * n
    dpB = [NEG_INF] * n

    # Backpointers hold the previous state symbol ("A","0","B") or None if unreachable.
    bpA = [None] * n
    bp0 = [None] * n
    bpB = [None] * n

    # ---- Base (i = 0) ----
    if sA[0] > NEG_INF:
        dpA[0] = sA[0]
        bpA[0] = "A"  # start-in-A

    dp0[0] = s0[0]
    bp0[0] = "0"  # start-in-Ø

    if sB[0] > NEG_INF:
        dpB[0] = dp0[0] + sB[0]
        bpB[0] = "0"

    # ---- Forward (i >= 1) ----
    for i in range(1, n):
        # A -> A
        if dpA[i - 1] > NEG_INF and sA[i] > NEG_INF:
            dpA[i] = dpA[i - 1] + sA[i]
            bpA[i] = "A"

        # Ø from best of (A, Ø)
        prevA = dpA[i - 1]
        prev0 = dp0[i - 1]
        if prevA <= NEG_INF and prev0 <= NEG_INF:
            dp0[i] = NEG_INF
            bp0[i] = None
        else:
            if prevA >= prev0:
                dp0[i] = prevA + s0[i]
                bp0[i] = "A"
            else:
                dp0[i] = prev0 + s0[i]
                bp0[i] = "0"

        # B from best of (Ø, B)
        prevB = dpB[i - 1]
        prev0 = dp0[i - 1]
        if sB[i] <= NEG_INF or (prevB <= NEG_INF and prev0 <= NEG_INF):
            dpB[i] = NEG_INF
            bpB[i] = None
        else:
            if prev0 >= prevB:
                dpB[i] = prev0 + sB[i]
                bpB[i] = "0"
            else:
                dpB[i] = prevB + sB[i]
                bpB[i] = "B"

    # ---- Choose terminal state ----
    best_val, state = max(
        (dpA[n - 1], "A"), (dp0[n - 1], "0"), (dpB[n - 1], "B"), key=lambda x: x[0]
    )
    if best_val <= NEG_INF:
        # Nothing was feasible → everything is Ø
        return ["0"] * n

    # ---- Backtrack robustly ----
    labels = ["0"] * n
    i = n - 1
    while i >= 0:
        labels[i] = state

        if state == "A":
            if bpA[i] is None:
                state = "0" if (dp0[i] > NEG_INF) else "A"
            else:
                state = bpA[i]
            i -= 1
        elif state == "0":
            if bp0[i] is None:
                if i > 0 and dp0[i - 1] > NEG_INF:
                    state = "0"
                elif i > 0 and dpA[i - 1] > NEG_INF:
                    state = "A"
                else:
                    state = "0"
            else:
                state = bp0[i]
            i -= 1
        else:  # "B"
            if bpB[i] is None:
                if i > 0 and dpB[i - 1] > NEG_INF:
                    state = "B"
                elif i > 0 and dp0[i - 1] > NEG_INF:
                    state = "0"
                else:
                    state = "0"
            else:
                state = bpB[i]
            i -= 1

    # Enforce A* 0* B* explicitly (safety)
    lastA = max([k for k, lb in enumerate(labels) if lb == "A"], default=-1)
    firstB = min([k for k, lb in enumerate(labels) if lb == "B"], default=n)
    for k in range(0, lastA + 1):
        labels[k] = "A"
    for k in range(firstB, n):
        labels[k] = "B"
    for k in range(lastA + 1, firstB):
        if labels[k] not in ("A", "B"):
            labels[k] = "0"

    return labels


def _refine_video(merged_timeline, args):
    """
    Refine all GTs in one video using optimal in-between assignment + one-sided edges.
    Returns dict key=(gt_start, gt_end) -> (new_start, new_end, caption)
    """
    gt_idxs = [i for i, s in enumerate(merged_timeline) if s["src"] == "gt"]
    if not gt_idxs:
        return {}

    result = {}
    for i in gt_idxs:
        g = merged_timeline[i]
        result[(g["start"], g["end"])] = [g["start"], g["end"], g["caption"]]

    # Pairwise in-between assignment (no crossing)
    for a_i, b_i in zip(gt_idxs, gt_idxs[1:]):
        A = merged_timeline[a_i]
        B = merged_timeline[b_i]
        between = [
            s
            for s in merged_timeline
            if (s["src"] == "pl" and s["start"] >= A["end"] and s["end"] <= B["start"])
        ]
        between.sort(key=lambda x: x["start"])
        if not between:
            continue

        sA, s0, sB = _compute_scores_and_masks(
            between, A, B, args.gap, args.use_only, args.alpha_nouns, args.beta_verbs
        )

        # Optional quick guard if both sides are entirely impossible
        if all(v <= -1e17 for v in sA) and all(v <= -1e17 for v in sB):
            labels = ["0"] * len(between)
        else:
            labels = _dp_assign_between(between, sA, s0, sB, args.tau_min)

        # Extend A to the right with A-labeled PLs
        a_new_end = result[(A["start"], A["end"])][1]
        for pl, lb in zip(between, labels):
            if lb == "A":
                a_new_end = max(a_new_end, pl["end"])
        result[(A["start"], A["end"])][1] = a_new_end

        # Extend B to the left with B-labeled PLs
        b_new_start = result[(B["start"], B["end"])][0]
        for pl, lb in zip(between, labels):
            if lb == "B":
                b_new_start = min(b_new_start, pl["start"])
        result[(B["start"], B["end"])][0] = b_new_start

    # Left edge: PLs before the first GT
    first_gt = merged_timeline[gt_idxs[0]]
    left_pls = [
        s for s in merged_timeline if s["src"] == "pl" and s["end"] <= first_gt["start"]
    ]
    left_pls.sort(key=lambda x: x["start"])
    if left_pls:
        reach = []
        for i in reversed(range(len(left_pls))):
            seg = left_pls[i]
            ok_overlap = _has_overlap(
                first_gt["nouns"],
                first_gt["verbs"],
                first_gt["noun_lemmas"],
                first_gt["verb_lemmas"],
                seg["nouns"],
                seg["verbs"],
                seg["noun_lemmas"],
                seg["verb_lemmas"],
                use_only=args.use_only,
            )
            if not ok_overlap:
                break
            if not reach:
                if abs(first_gt["start"] - seg["end"]) <= args.gap:
                    reach.append(i)
                else:
                    break
            else:
                prev_i = reach[-1]
                if abs(left_pls[prev_i]["start"] - seg["end"]) <= args.gap:
                    reach.append(i)
                else:
                    break
        if reach:
            i_min = min(reach)
            result[(first_gt["start"], first_gt["end"])][0] = min(
                result[(first_gt["start"], first_gt["end"])][0],
                left_pls[i_min]["start"],
            )

    # Right edge: PLs after the last GT
    last_gt = merged_timeline[gt_idxs[-1]]
    right_pls = [
        s for s in merged_timeline if s["src"] == "pl" and s["start"] >= last_gt["end"]
    ]
    right_pls.sort(key=lambda x: x["start"])
    if right_pls:
        reach = []
        for i in range(len(right_pls)):
            seg = right_pls[i]
            ok_overlap = _has_overlap(
                last_gt["nouns"],
                last_gt["verbs"],
                last_gt["noun_lemmas"],
                last_gt["verb_lemmas"],
                seg["nouns"],
                seg["verbs"],
                seg["noun_lemmas"],
                seg["verb_lemmas"],
                use_only=args.use_only,
            )
            if not ok_overlap:
                break
            if not reach:
                if abs(seg["start"] - last_gt["end"]) <= args.gap:
                    reach.append(i)
                else:
                    break
            else:
                prev_i = reach[-1]
                if abs(seg["start"] - right_pls[prev_i]["end"]) <= args.gap:
                    reach.append(i)
                else:
                    break
        if reach:
            i_max = max(reach)
            result[(last_gt["start"], last_gt["end"])][1] = max(
                result[(last_gt["start"], last_gt["end"])][1], right_pls[i_max]["end"]
            )

    return result


def main(args):
    wandb.init(
        project="Thesis",
        name=f"Refine Dataset - Gap {args.gap}s",
        config={**args.__dict__},
        group=f"Refine Dataset",
    )

    ground_truth_data = load_data(args.gt)
    pseudo_labels_data = load_data(args.pseudolabels)

    total = len(ground_truth_data)
    total_pl = len(pseudo_labels_data)
    print(f"Total ground truth segments: {total}")
    print(f"Total pseudo-labels segments: {total_pl}")

    # Group by video_id
    print("Grouping videos by video_id...")
    ground_truth_video_groups = _group_videos_by_video_id(ground_truth_data)
    pseudo_labels_video_groups = _group_videos_by_video_id(pseudo_labels_data)

    # Build tagged merged timelines
    merged = _build_tagged_timeline(
        ground_truth_video_groups, pseudo_labels_video_groups
    )

    # Refine per video using optimal assignments (no GT crossing)
    refined_map = {}  # (video_id, gt_start, gt_end) -> (new_start, new_end, caption)
    for vid, timeline in tqdm(merged.items(), desc="Refining per video"):
        vid_result = _refine_video(timeline, args)
        for (gt_start, gt_end), (new_s, new_e, cap) in vid_result.items():
            refined_map[(vid, gt_start, gt_end)] = (new_s, new_e, cap)

    # Emit refined_data in original GT order
    refined_data = []
    for (
        video_id,
        gt_start,
        gt_end,
        gt_original_caption,
        *_extras,
    ) in ground_truth_data:
        key = (video_id, float(gt_start), float(gt_end))
        if key in refined_map:
            new_start, new_end, _cap = refined_map[key]
            refined_data.append((video_id, new_start, new_end, gt_original_caption))
        else:
            refined_data.append(
                (video_id, float(gt_start), float(gt_end), gt_original_caption)
            )

    save_data(f"{args.out_path}/ego4d_train_refined_gap_{args.gap}.pkl", refined_data)

    # --- Plots & summaries (unchanged from your version) ---
    old_durations = [
        float(end) - float(start) for (_, start, end, *_) in ground_truth_data
    ]
    new_durations = [float(end) - float(start) for (_, start, end, _) in refined_data]

    old = np.asarray(old_durations, dtype=float)
    new = np.asarray(new_durations, dtype=float)
    delta = new - old

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.hist(old, bins=100, alpha=0.6, label="Old")
    ax.hist(new, bins=100, alpha=0.6, label="New")
    ax.set_yscale("log")
    ax.set_xlabel("Segment length (seconds)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Old vs New Segment Length Distribution (gap={args.gap}s)")
    ax.legend()
    wandb.log({"plots/length_distribution": wandb.Image(fig)})
    plt.close(fig)

    def safe_std(x):
        return float(x.std(ddof=1)) if x.size > 1 else 0.0

    summaries = {
        "counts/total": int(old.size),
        "changes/num_expanded": int((delta > 0).sum()),
        "changes/num_unchanged": int((delta == 0).sum()),
        "old/mean": float(old.mean()),
        "old/median": float(np.median(old)),
        "old/std": safe_std(old),
        "old/min": float(old.min()) if old.size else 0.0,
        "old/max": float(old.max()) if old.size else 0.0,
        "old/p25": float(np.percentile(old, 25)) if old.size else 0.0,
        "old/p75": float(np.percentile(old, 75)) if old.size else 0.0,
        "new/mean": float(new.mean()),
        "new/median": float(np.median(new)),
        "new/std": safe_std(new),
        "new/min": float(new.min()) if new.size else 0.0,
        "new/max": float(new.max()) if new.size else 0.0,
        "new/p25": float(np.percentile(new, 25)) if new.size else 0.0,
        "new/p75": float(np.percentile(new, 75)) if new.size else 0.0,
        "delta/mean": float(delta.mean()),
        "delta/median": float(np.median(delta)),
        "delta/p95": float(np.percentile(delta, 95)) if delta.size else 0.0,
    }
    wandb.log(summaries)
    wandb.log(
        {
            "hist/old": wandb.Histogram(old),
            "hist/new": wandb.Histogram(new),
            "hist/delta": wandb.Histogram(delta),
        }
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
