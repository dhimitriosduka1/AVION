import pickle
import os

path = "/dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_42.pkl"

# 1. Load the original data
print(f"Loading data from {path}...")
with open(path, "rb") as f:
    data = pickle.load(f)

print(f"Original data size (samples): {len(data)}")

# 2. Group samples by video_id
video_dict = {}
for sample in data:
    video_id = sample[0]
    if video_id not in video_dict:
        video_dict[video_id] = []
    video_dict[video_id].append(sample)

print(f"Number of unique videos: {len(video_dict)}")

# 3. Sort keys and select the first 1000
sorted_video_ids = sorted(video_dict.keys())
target_video_ids = sorted_video_ids[:1000]

# Write the target_video_ids to a file
with open("target_video_ids.txt", "w") as f:
    for video_id in target_video_ids:
        f.write(f"{video_id}\n")

print(f"Selected {len(target_video_ids)} videos for debug set.")

# 4. Flatten the selected videos back into a list of samples
debug_data = []
for video_id in target_video_ids:
    debug_data.extend(video_dict[video_id])

print(f"Debug data size (samples): {len(debug_data)}")

# 5. Save to new pickle file
output_path = path.replace(".pkl", "_debug.pkl")

with open(output_path, "wb") as f:
    pickle.dump(debug_data, f)

print(f"Saved debug dataset to: {output_path}")
