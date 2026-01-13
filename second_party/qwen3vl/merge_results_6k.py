import os
import json
import pickle

EGO4D_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl"
SCALED_PATH = "/dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_42.pkl"
BASE_PATH = "/dais/fs/scratch/dduka/databases/ego4d/debug_split/6k"

json_files = [f for f in os.listdir(BASE_PATH) if f.endswith(".json")]

video_ids = [jf.split("_")[1].split(".")[0] for jf in json_files]

# with open(EGO4D_PATH, "rb") as f:
#     ego4d_data = pickle.load(f)

# with open(SCALED_PATH, "rb") as f:
#     scaled_data = pickle.load(f)

# filtered_ego4d = [s for s in ego4d_data if s[0] in video_ids]
# scaled_filteres = [s for s in scaled_data if s[0] in video_ids]

# print(f"Total samples {len(filtered_ego4d)} filtered from Ego4D")
# print(f"Total samples {len(scaled_filteres)} filtered from Scaled")

results = []
nr_error = 0
nr_other = 0
for json_file in json_files:
    with open(os.path.join(BASE_PATH, json_file), "r") as f:
        json_data = json.load(f)

        for sample in json_data:
            if "error" in sample:
                nr_error += 1
                continue

            if sample["pred_start_global"] == sample["pred_end_global"]:
                results.append(
                    (
                        sample["video_id"],
                        sample["base_offset"] + sample["seed_start_rel"],
                        sample["base_offset"] + sample["seed_end_rel"],
                        sample["caption"],
                    )
                )
                nr_other += 1
            else:
                results.append(
                    (
                        sample["video_id"],
                        sample["pred_start_global"],
                        sample["pred_end_global"],
                        sample["caption"],
                    )
                )

print(f"Number of errors: {nr_error}")
print(f"Number of others: {nr_other}")

# with open(os.path.join(BASE_PATH, "ego4d_train_6k_debug_split.pkl"), "wb") as f:
#     pickle.dump(filtered_ego4d, f)
#     print(f"Wrote {len(filtered_ego4d)} into ego4d_train_6k_debug_split.pkl")

# with open(os.path.join(BASE_PATH, "scaled_2.1_6k_debug_split.pkl"), "wb") as f:
#     pickle.dump(scaled_filteres, f)
#     print(f"Wrote {len(scaled_filteres)} into scaled_2.1_6k_debug_split.pkl ")

with open(os.path.join(BASE_PATH, "qwen_refined_6k_debug_split.pkl"), "wb") as f:
    pickle.dump(results, f)
    print(f"Wrote {len(results)} into qwen_refined_6k_debug_split.pkl ")
