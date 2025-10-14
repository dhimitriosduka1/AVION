import json
from pathlib import Path
from torch.utils.data import Dataset


class VideoMetadataDataset(Dataset):
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.root = Path(self.metadata_path)

        print(f"Resolving metadata paths")
        self.metadata_paths = list(self.root.rglob("*.json"))
        print(f"Found {len(self.metadata_paths)} metadata files")

        self.captions = []
        for metadata_path in self.metadata_paths:
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)

            for captions in metadata_dict["metadata"]:
                cleaned = [c.rstrip(". \t\r\n") for c in captions["captions"]]
                self.captions.extend(cleaned)

        captions_length = len(self.captions)
        unique_captions_length = len(list(set(self.captions)))

        print(f"All captions: {captions_length}")
        print(f"Unique captions: {unique_captions_length}")

        print(
            f"Percentage of captions that are unique: {unique_captions_length / captions_length}"
        )

    def __len__(self):
        return len(self.metadata_paths)

    def __getitem__(self, idx):
        metadata_path = self.metadata_paths[idx]

        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        metadata = metadata_dict["metadata"]

        return metadata


if __name__ == "__main__":
    import time

    start_time = time.time()
    dataset = VideoMetadataDataset(
        metadata_path="/ptmp/dduka/databases/ego4d/video_320px_15sec/lavila_captions_num_frames_4/temperature_1.0"
    )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
