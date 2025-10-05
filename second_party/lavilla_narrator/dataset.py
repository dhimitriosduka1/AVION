import os
import torch
import numpy as np

from lavila.data.datasets import get_frames


class VideoNarratorDataset(torch.utils.data.Dataset):
    """
    Dataset for loading videos and their generated captions.
    """

    def __init__(
        self,
        video_root,
        caption_suffix,
        num_frames=4,
        num_segments=60,
        val_transform=None,
        jitter=False,
    ):
        self.video_root = video_root
        self.caption_suffix = caption_suffix
        self.num_frames = num_frames
        self.num_segments = num_segments
        self.val_transform = val_transform
        self.jitter = jitter

        self.number_of_chunks = self.num_segments // self.num_frames
        assert self.number_of_chunks == 15

        self.samples = self._load_samples()

    def _load_samples(self):
        """Load all video paths and their corresponding caption files."""
        samples = []

        for dirpath, _, filenames in os.walk(self.video_root):
            for name in filenames:
                if name.lower().endswith(".mp4"):
                    video_path = os.path.abspath(os.path.join(dirpath, name))

                    # Construct caption path
                    caption_path = video_path.replace(
                        "/video_320px_15sec/",
                        f"/video_320px_15sec/{self.caption_suffix}/",
                    )

                    samples.append(
                        {
                            "video_path": video_path,
                            "caption_path": caption_path,
                        }
                    )

        return samples

    def _load_frames(self, video_path):
        original_frames, frame_ids, fps = get_frames(
            video_path=video_path, num_segments=self.num_segments, jitter=self.jitter
        )

        original_frames_chunked = original_frames.chunk(self.number_of_chunks)
        frame_ids_chunked = np.array_split(frame_ids, self.number_of_chunks)

        frames_chunked = []
        for chunk in original_frames_chunked:
            frames_chunked.append(self.val_transform(chunk).unsqueeze(0))

        return frames_chunked, frame_ids_chunked, fps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor of shape (C, T, H, W) if transform is applied
            captions: String or list of strings depending on return_all_captions
            metadata: Dictionary with additional information (frame_ids, timestamps, fps, etc.)
        """
        sample = self.samples[idx]

        video_path = sample["video_path"]
        caption_path = sample["caption_path"]

        frames, frame_ids, fps = self._load_frames(video_path)

        return {
            "video_path": video_path,
            "frames": frames,
            "frames_ids": frame_ids,
            "fps": fps,
            "caption_path": caption_path,
        }
