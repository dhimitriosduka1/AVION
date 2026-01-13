# Create a .pkl file with the results from the json files

import os
import json
import pickle

original_pkl_file = (
    "/dais/fs/scratch/dduka/databases/ego4d/debug_split/ego4d_train_debug.pkl"
)
scaled_pkl_file = "/dais/fs/scratch/dduka/databases/ego4d/debug_split/ego4d_train_random_shift_2.1_2.1_1.0_42_debug.pkl"
base_dir = "/u/dduka/project/AVION/second_party/qwen3vl/output_scaled"

# Compute original segments
with open(original_pkl_file, "rb") as f:
    original_segments = pickle.load(f)
    original_segment_lens = [s[2] - s[1] for s in original_segments]

print(f"Original segments: {len(original_segments)}")

# Compute scaled segments
with open(scaled_pkl_file, "rb") as f:
    scaled_segments = pickle.load(f)
    scaled_segment_lens = [s[2] - s[1] for s in scaled_segments]

print(f"Scaled segments: {len(scaled_segments)}")

# Get all the json files in the base directory
json_files = [f for f in os.listdir(base_dir) if f.endswith(".json")]


# ====================

max_len = -1
max_len_video_id = ""
for json_file in json_files:
    with open(os.path.join(base_dir, json_file), "r") as f:
        json_data = json.load(f)
        if len(json_data) > max_len:
            max_len = len(json_data)
            max_len_video_id = json_data[0]["video_id"]

print(f"Max len: {max_len}")
print(f"Max len video id: {max_len_video_id}")

exit()
# ====================


broken = 0
equal = 0
qwen3vl_segments = []
qwen3vl_data = []
for json_file in json_files:
    with open(os.path.join(base_dir, json_file), "r") as f:
        json_data = json.load(f)
        for sample in json_data:
            try:
                if sample["pred_start_global"] == sample["pred_end_global"]:
                    qwen3vl_segments.append(
                        (
                            sample["video_id"],
                            sample["base_offset"] + sample["seed_start_rel"],
                            sample["base_offset"] + sample["seed_end_rel"],
                            sample["caption"],
                        )
                    )
                else:
                    qwen3vl_data.append(
                        (
                            sample["video_id"],
                            sample["pred_start_global"],
                            sample["pred_end_global"],
                            sample["caption"],
                        )
                    )

                qwen3vl_segments.append(
                    sample["pred_end_global"] - sample["pred_start_global"]
                )

            except KeyError:
                broken += 1
                continue

print(f"Broken: {broken}")
print(f"Equal: {equal}")
print(f"Qwen3vl segments: {len(qwen3vl_segments)}")

print(f"Saving qwen3vl data to qwen3vl_data.pkl")
with open(
    "/dais/fs/scratch/dduka/databases/ego4d/debug_split/qwen3vl_data.pkl", "wb"
) as f:
    pickle.dump(qwen3vl_data, f)

# # Plot the segments
# import matplotlib.pyplot as plt

# plt.hist(original_segment_lens, bins=100, alpha=0.5, label="Original")
# plt.hist(scaled_segment_lens, bins=100, alpha=0.5, label="Scaled")
# plt.hist(qwen3vl_segments, bins=100, alpha=0.5, label="Qwen3vl")
# plt.legend()
# plt.show()
# plt.yscale("log")

# # Save the image
# plt.savefig("qwen3vl_segments.png")
