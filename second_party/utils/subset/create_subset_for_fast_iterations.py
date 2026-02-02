import random
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
OUTPUT_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset.pkl"
)
MAX_SAMPLES_PER_VIDEO = 50

random.seed(42)

with open(DATA_PATH, "rb") as f:
    data = pkl.load(f)

# Group by video_id to ensure unique videos
video_dict = defaultdict(list)
for item in data:
    video_dict[item[1]].append(item)

print(f"Total unique videos: {len(video_dict)}")

subset = []
below_threshold_count = 0
for video_id, items in video_dict.items():
    if len(items) > MAX_SAMPLES_PER_VIDEO:
        # Sort by timestamp index first
        items.sort(key=lambda x: x[2])

        # Divide into segments
        seg_size = len(items) // MAX_SAMPLES_PER_VIDEO
        sampled_items = []
        for i in range(MAX_SAMPLES_PER_VIDEO):
            segment = items[i * seg_size : (i + 1) * seg_size]
            if segment:
                sampled_items.append(random.choice(segment))
    else:
        sampled_items = items
        below_threshold_count += 1

    subset.extend(sampled_items)

print(f"Total samples in subset: {len(subset)}")
print(f"Total unique videos in subset: {len(set([s[1] for s in subset]))}")
print(f"Videos below threshold: {below_threshold_count}")

# Plot the distribution of samples per video in the subset
orginal_segment_lengths = [s[3] - s[2] for s in data]
subset_segment_lengths = [s[3] - s[2] for s in subset]
samples_per_video = [len(items) for items in video_dict.values()]

# Plot the original distribution
plt.figure(figsize=(10, 6))
plt.hist(orginal_segment_lengths, bins=50, color="green", alpha=0.7)
plt.title("Distribution of Segment Lengths in Original Data")
plt.xlabel("Segment Length (frames)")
plt.ylabel("Number of Samples")
plt.yscale("log")
plt.grid(True)
plt.savefig(
    "/u/dduka/project/AVION/images/subset_original_segment_length_distribution.png"
)
plt.close()

# Plot the subset distribution
plt.figure(figsize=(10, 6))
plt.hist(subset_segment_lengths, bins=50, color="blue", alpha=0.7)
plt.title("Distribution of Segment Lengths in Subset")
plt.xlabel("Segment Length (frames)")
plt.ylabel("Number of Samples")
plt.yscale("log")
plt.grid(True)
plt.savefig("/u/dduka/project/AVION/images/subset_segment_length_distribution.png")
plt.close()

# Plot samples per video
plt.figure(figsize=(10, 6))
plt.hist(
    samples_per_video,
    bins=50,
    color="orange",
    alpha=0.7,
)
plt.title("Number of Samples per Video in Original Data")
plt.xlabel("Number of Samples")
plt.ylabel("Number of Videos")
plt.yscale("log")
plt.grid(True)
plt.savefig("/u/dduka/project/AVION/images/subset_samples_per_video_distribution.png")
plt.close()

# Save the subset to a pickle file
with open(
    "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset_with_uuid.pkl",
    "wb",
) as f:
    pkl.dump(subset, f)

with open(
    "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset.pkl", "wb"
) as f:
    pkl.dump([(s[1], s[2], s[3], s[4]) for s in subset], f)
