import pickle as pkl
from tqdm import tqdm

EGO4D_ORIGINAL_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
LAVILA_REPHRASED_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3.pkl"
)
OUTPUT_LAVILA_REPHRASED_WITH_UUID_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_with_uuid.pkl"

with open(EGO4D_ORIGINAL_PATH, "rb") as f:
    original_data = pkl.load(f)

with open(LAVILA_REPHRASED_PATH, "rb") as f:
    lavila_data = pkl.load(f)

results = []

for i, (o_sample, l_sample) in tqdm(
    enumerate(zip(original_data, lavila_data[: (len(original_data))]))
):
    assert len(o_sample) == 5 and len(l_sample) == 4
    assert o_sample[2] == l_sample[1]
    assert o_sample[3] == l_sample[2]
    assert o_sample[4] == l_sample[3][0]

    results.append(
        # uuid, video_id, start, end, captions
        (o_sample[0], l_sample[0], l_sample[1], l_sample[2], l_sample[3])
    )

print(f"Missing elements: {lavila_data[(len(original_data)):]}")

with open(OUTPUT_LAVILA_REPHRASED_WITH_UUID_PATH, "wb") as f:
    pkl.dump(results, f)
