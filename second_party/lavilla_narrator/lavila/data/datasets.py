import torch
import decord
import numpy as np
import os.path as osp


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


# This is just for understanding the code, not used in the project
if __name__ == "__main__":
    frame_ids = get_frame_ids(0, 450, num_segments=15, jitter=False)
    print(f"Frame IDs using number of segments 15: {frame_ids}")

    frame_ids = get_frame_ids(0, 450, num_segments=60)
    print(f"Frame IDs using number of segments 30: {frame_ids}")
