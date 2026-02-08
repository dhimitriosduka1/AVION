import pickle as pkl
from tqdm import tqdm

BASE_SUBSET = "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset_with_uuid.pkl"
FULL_REFINED = "/dais/fs/scratch/dduka/databases/ego4d/qwen_refinement/scaled/pickle/ego4d_train_scaled_10_caption_vllm_with_uuid.pkl"

with open(BASE_SUBSET, "rb") as f:
    data = pkl.load(f)

with open(FULL_REFINED, "rb") as f:
    full_refined_data = pkl.load(f)

row_uuids = {s[0]: True for s in data}
subset = [s for s in tqdm(full_refined_data) if s[0] in row_uuids]

print(f"Saving {len(subset)} in file...")

with open(
    "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset_scaled_10_cap_with_uuid.pkl",
    "wb",
) as f:
    pkl.dump(subset, f)

with open(
    "/dais/fs/scratch/dduka/databases/ego4d/subset/ego4d_train_362k_subset_scaled_10_cap.pkl", "wb"
) as f:
    pkl.dump([(s[1], s[2], s[3], s[4]) for s in subset], f)
