import torch
import decord
import numpy as np
import os

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    """
    Get frame ids from a video reader.
    Args:
        start_frame: start frame
        end_frame: end frame
        num_segments: number of segments
        jitter: whether to jitter the frames
    Returns:
        frame_ids: list of integers
        if jitter is True, the frame ids are random, otherwise they are the middle of the segments
    """
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def video_loader_by_frames(video_reader, frame_ids):
    """
    Load frames from a video reader by frame ids.
    Args:
        video_reader: decord.VideoReader
        frame_ids: list of integers
    Returns:
        frames: tensor of frames
    """
    try:
        frames = video_reader.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", video_reader)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


def get_frames(video_path, num_segments, jitter=False):
    """
    Get frames from a video. This method assummes that the frames are always sampled between the start and end of the video.
    Args:
        video_path: path to the video
        num_segments: number of segments
        jitter: whether to jitter the frames
    Returns:
        frames: frames from the video
        frame_ids: frame ids used to load the frames
    """
    video_reader = decord.VideoReader(video_path)

    frame_ids = get_frame_ids(
        0, len(video_reader), num_segments=num_segments, jitter=jitter
    )

    fps = video_reader.get_avg_fps()
    return video_loader_by_frames(video_reader, frame_ids), frame_ids, fps


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
            frames_chunked.append(self.val_transform(chunk))

        return frames_chunked, frame_ids_chunked, fps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx: index of the sample
        Returns:
            frames: Tensor of shape (T, H, W, C)
            metadata: Dictionary with additional information (frame_ids, timestamps, fps, etc.)
        """
        sample = self.samples[idx]

        video_path = sample["video_path"]
        caption_path = sample["caption_path"]

        frames, frame_ids, fps = self._load_frames(video_path)

        return {
            "video_path": video_path,
            "frames": torch.stack(frames, dim=0),
            "frames_ids": torch.tensor(np.array(frame_ids)),
            "fps": torch.tensor(np.array(fps)),
            "caption_path": caption_path,
        }
