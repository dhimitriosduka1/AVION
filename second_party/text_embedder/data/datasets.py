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
        print(
            f"Percentage of unique captions: {self.percentage_of_unique_captions}")
        print(f"Preprocess function: {self.preprocess_function}")

    def set_custom_tokenizer_function(self, tokenizer_function):
        self.custom_tokenizer_function = tokenizer_function

    def __len__(self):
        return len(self.unique_captions)

    def __getitem__(self, idx):
        metadata = self.unique_captions[idx]

        original_caption = metadata["text"]

        if self.custom_tokenizer_function is not None:
            caption = self.custom_tokenizer_function(original_caption)
        else:
            # Tokenize as a single example and remove the leading batch dim so
            # DataLoader batching produces shape [batch, context_length]
            caption = self.tokenizer([original_caption])[0]

        frequency = metadata["frequency"]

        return {
            "original_caption": original_caption,
            "caption": caption,
            "frequency": torch.tensor(frequency)
        }
