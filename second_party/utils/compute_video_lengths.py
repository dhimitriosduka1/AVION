import os
import cv2
import json
import argparse
from tqdm import tqdm

DEFAULT_VIDEO_ROOT = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/"
DEFAULT_OUTPUT_FILE_PATH = "/dais/fs/scratch/dduka/databases/ego4d/video_lengths.json"


def get_video_duration(file_path):
    """
    Opens a video file and calculates its duration in seconds.
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = 0.0
        if fps > 0:
            duration = frame_count / fps

        cap.release()
        return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute lengths of full video files.")
    parser.add_argument(
        "--video_root",
        type=str,
        default=DEFAULT_VIDEO_ROOT,
        help="Directory containing the .mp4 video files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_FILE_PATH,
        help="Path to save the output JSON.",
    )

    args = parser.parse_args()

    video_lengths = {}
    files_to_process = []

    print(f"Scanning {args.video_root} for .mp4 files...")

    try:
        files = os.listdir(args.video_root)
        for f in files:
            if f.endswith(".mp4"):
                files_to_process.append(os.path.join(args.video_root, f))
    except FileNotFoundError:
        print(f"Error: Directory {args.video_root} not found.")
        return

    print(f"Found {len(files_to_process)} videos. Computing lengths...")

    for file_path in tqdm(files_to_process):
        # Extract ID: filename without extension
        # e.g., "/path/to/video123.mp4" -> "video123"
        filename = os.path.basename(file_path)
        video_id = os.path.splitext(filename)[0]

        duration = get_video_duration(file_path)

        if duration > 0:
            video_lengths[video_id] = duration
        else:
            print(f"Warning: Video {video_id} has 0 duration.")

    print(f"Successfully computed lengths for {len(video_lengths)} videos.")
    print(f"Saving to {args.output_path}...")

    with open(args.output_path, "w") as f:
        json.dump(video_lengths, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
