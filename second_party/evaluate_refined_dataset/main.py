import pickle as pkl
import pandas as pd
from collections import defaultdict

EGO4D_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl"
EGO4D_SCALED_PATH = "/dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_42.pkl"
EGO4D_QWEN_REFINED_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/debug_split/1k/qwen3vl_data.pkl"
)
REFERENCE_PATH = (
    "/u/dduka/project/AVION/second_party/evaluate_refined_dataset/data/segments.csv"
)
REFERENCE_VIDEO_IDS_PATH = (
    "/u/dduka/project/AVION/second_party/evaluate_refined_dataset/data/video_ids.csv"
)


def compute_iou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2

    start_inter = max(s1, s2)
    end_inter = min(e1, e2)

    intersection = max(0, end_inter - start_inter)

    duration1 = e1 - s1
    duration2 = e2 - s2

    union = duration1 + duration2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def group_videos_by_id(data):
    grouped = defaultdict(list)

    for sample in data:
        grouped[sample[0]].append(sample)

    for video_id in grouped.keys():
        grouped[video_id].sort(key=lambda x: x[1])

    return grouped


def resolve_reference_data():
    segments_df = pd.read_csv(REFERENCE_PATH)
    video_ids = pd.read_csv(REFERENCE_VIDEO_IDS_PATH)

    target_ids = video_ids["video_id"]
    return group_videos_by_id(
        list(
            segments_df[segments_df["video_id"].isin(target_ids)].itertuples(
                index=False, name=None
            )
        )
    )


def eval_alignment(refined_data, gt_data):
    iou_results = []

    for video_id, gt_segments in gt_data.items():
        # FIXME
        if video_id in [
            "022a0833-bd07-4dc6-8b94-2d82f0881009",
            # "068c3c6b-fbbc-486f-bc4c-f56e26448a09",
            # "0ed57e80-0e57-47d3-8942-54450722dc95",
        ]:
            continue

        refined_segments = refined_data[video_id]
        assert len(refined_segments) == len(gt_segments)

        # Construct a dict where the key is the caption
        gt_map = defaultdict(list)
        for sample in gt_segments:
            gt_map[sample[3]].append(sample)

        for refined_segment in refined_segments:
            print(refined_segment[3])
            gt_segment = gt_map[refined_segment[3]].pop(0)
            iou_results.append(
                compute_iou(
                    (gt_segment[1], gt_segment[2]),
                    (refined_segment[1], refined_segment[2]),
                )
            )

    return sum(iou_results) / len(iou_results)


with open(EGO4D_PATH, "rb") as f:
    ego4d_data = group_videos_by_id(pkl.load(f))

with open(EGO4D_SCALED_PATH, "rb") as f:
    ego4d_scaled_data = group_videos_by_id(pkl.load(f))

with open(EGO4D_QWEN_REFINED_PATH, "rb") as f:
    ego4d_qwen_refined_data = group_videos_by_id(pkl.load(f))

reference_data = resolve_reference_data()

eval_alignment(ego4d_data, reference_data)
