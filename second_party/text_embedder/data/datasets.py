import json
import torch
from torch.utils.data import Dataset


class VideoMetadataDataset(Dataset):
    """
    Dataset for loading video metadata.
    The metadata is a json file with the following structure:
    {
        "number_of_total_captions": int,
        "number_of_unique_captions": int,
        "percentage_of_unique_captions": float,
        "unique_captions": [
            {
                "text": str,
                "frequency": int,
            }
        ],
        "preprocess_function": {
            "source": str,
        },
    }
    """

    def __init__(self, metadata_path, tokenizer):
        # Assert the file is a json file
        assert metadata_path.endswith(".json"), "The file must be a json file"

        self.metadata_path = metadata_path
        self.tokenizer = tokenizer

        assert tokenizer is not None, "Tokenizer is required"

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.number_of_total_captions = self.metadata["number_of_total_captions"]
        self.number_of_unique_captions = self.metadata["number_of_unique_captions"]
        self.percentage_of_unique_captions = self.metadata[
            "percentage_of_unique_captions"
        ]
        self.unique_captions = self.metadata["unique_captions"]
        self.preprocess_function = self.metadata["preprocess_function"]

        print(f"Number of total captions: {self.number_of_total_captions}")
        print(f"Number of unique captions: {self.number_of_unique_captions}")
        print(f"Percentage of unique captions: {self.percentage_of_unique_captions}")
        print(f"Preprocess function: {self.preprocess_function}")

    def __len__(self):
        return len(self.unique_captions)

    def __getitem__(self, idx):
        metadata = self.unique_captions[idx]

        caption = metadata["text"]

        # Make sure this is okay
        caption = self.tokenizer(caption)

        frequency = metadata["frequency"]

        return caption, torch.tensor(frequency)
