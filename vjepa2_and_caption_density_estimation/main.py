import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel

HF_REPO = "facebook/vjepa2-vitg-fpc64-384"

WINDOW_SIZE = 64
WINDOW_STRIDE = 8
FRAME_STRIDE = 4
BATCH_SIZE = 128

def load_model_and_processor():
    model = AutoModel.from_pretrained(HF_REPO, dtype=torch.float16)
    model.to("cuda")
    model.eval()

    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    return model, processor


def create_windows(video_length, window_size, stride, fps):
    frame_idx = np.arange(0, video_length, fps)

    windows = []

    for i in range(0, len(frame_idx) - window_size + 1, stride):
        windows.append(frame_idx[i : i + window_size])

    if windows[-1][-1] < frame_idx[-1]:
        windows.append(frame_idx[-window_size:])

    return windows, len(frame_idx)


def spatial_pooling(video_embeddings, num_frames):
    B, num_tokens, D = video_embeddings.shape
    num_spatial = num_tokens // num_frames
    video_embeddings = video_embeddings.view(B, num_frames, num_spatial, D)
    return video_embeddings.mean(dim=2)


def process_video(video_path, model, processor, output_dir):
    vr = VideoDecoder(video_path)
    num_raw_frames = len(vr)
    print(f"  Video has {num_raw_frames} raw frames.")

    windows, num_sampled_frames = create_windows(
        num_raw_frames, WINDOW_SIZE, WINDOW_STRIDE, FRAME_STRIDE
    )
    print(f"  Created {len(windows)} windows, {num_sampled_frames} sampled frames.")

    hidden_size = model.config.hidden_size

    # Accumulators for overlap averaging
    frame_accum = torch.zeros(num_sampled_frames, hidden_size)
    frame_count = torch.zeros(num_sampled_frames)

    # Build a mapping from raw frame index → sampled frame index
    sampled_indices = np.arange(0, num_raw_frames, FRAME_STRIDE)
    raw_to_sampled = {
        raw_idx: sampled_idx for sampled_idx, raw_idx in enumerate(sampled_indices)
    }

    with torch.no_grad():
        for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="  Processing Batches"):
            batch_windows = windows[i : i + BATCH_SIZE]
            batch_videos = []

            for window in batch_windows:
                video = vr.get_frames_at(indices=window).data
                batch_videos.append(video)

            inputs = processor(batch_videos, return_tensors="pt").to(model.device)

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            video_embeddings = model.get_vision_features(**inputs)

            # Spatial average pooling: (B, T*S, D) -> (B, T, D)
            per_frame = spatial_pooling(video_embeddings, WINDOW_SIZE).cpu().float()

            # Accumulate into full-video tensors
            for j, window in enumerate(batch_windows):
                for t, raw_idx in enumerate(window):
                    sampled_idx = raw_to_sampled[raw_idx]
                    frame_accum[sampled_idx] += per_frame[j, t]
                    frame_count[sampled_idx] += 1

    # Average overlapping frames
    frame_count = frame_count.clamp(min=1)
    video_embedding = frame_accum / frame_count.unsqueeze(1)

    # Save
    stem = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{stem}.pth")
    torch.save(video_embedding, output_path)
    print(f"  Saved {output_path} with shape {video_embedding.shape}")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Temporal Segmentation")
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save .pth embeddings",
    )
    parser.add_argument(
        "--ext", type=str, default="mp4", help="Video file extension (default: mp4)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_model_and_processor()

    video_files = sorted(Path(args.video_dir).glob(f"*.{args.ext}"))
    print(f"Found {len(video_files)} video(s) in {args.video_dir}")

    for idx, video_path in enumerate(video_files):
        print(f"\n[{idx + 1}/{len(video_files)}] Processing: {video_path.name}")

        output_path = os.path.join(args.output_dir, f"{video_path.stem}.pth")
        if os.path.exists(output_path):
            print(f"  Skipping (already exists): {output_path}")
            continue

        process_video(str(video_path), model, processor, args.output_dir)


if __name__ == "__main__":
    main()
