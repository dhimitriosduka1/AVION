import pickle as pkl
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Merge refined timestamps with LaViLa rephrased captions."
    )

    # Paths as default values
    parser.add_argument(
        "--source_timestamps",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/random_shift/ego4d_train_random_shift_2.1_2.1_1.0_with_uuid.pkl",
        help="Path to source timestamps pkl",
    )

    parser.add_argument(
        "--lavila_rephrased",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_with_uuid.pkl",
        help="Path to LaViLa rephrased pkl",
    )

    parser.add_argument(
        "--out_with_uuid",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_refined_with_uuid.pkl",
        help="Output path for data with UUID",
    )

    parser.add_argument(
        "--out_refined",
        type=str,
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.rephraser.no_punkt_top3_refined.pkl",
        help="Output path for refined data without UUID",
    )

    args = parser.parse_args()

    # Load Data
    print(f"Loading: {args.source_timestamps}")
    with open(args.source_timestamps, "rb") as f:
        refined_data = pkl.load(f)
        assert len(refined_data[0]) == 5

    print(f"Loading: {args.lavila_rephrased}")
    with open(args.lavila_rephrased, "rb") as f:
        lavila_data = pkl.load(f)
        assert len(lavila_data[0]) == 5

    # Create lookups via dict comprehension
    refined_dict = {sample[0]: sample for sample in refined_data}
    lavila_dict = {sample[0]: sample for sample in lavila_data}

    results = []
    results_without_uuid = []

    # Merge logic
    for key, sample in lavila_dict.items():
        if key in refined_dict:
            refined_sample = refined_dict[key]
            # (UUID, Lavila_1, Refined_2, Refined_3, Lavila_4)
            merged = (
                sample[0],
                sample[1],
                refined_sample[2],
                refined_sample[3],
                sample[4],
            )
            results.append(merged)
            results_without_uuid.append(merged[1:])

    print(f"Merged {len(results)} samples.")

    # Save outputs
    with open(args.out_with_uuid, "wb") as f:
        pkl.dump(results, f)

    with open(args.out_refined, "wb") as f:
        pkl.dump(results_without_uuid, f)

    print("Files saved successfully.")


if __name__ == "__main__":
    main()
