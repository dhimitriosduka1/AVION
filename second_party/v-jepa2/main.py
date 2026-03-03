import os
import torch
import h5py
import numpy as np
import subprocess
import tempfile
from decord import VideoReader, cpu
from transformers import AutoModel, AutoVideoProcessor
from accelerate import Accelerator
from tqdm import tqdm

# --- GLOBAL CONFIGURATION ---
WINDOW_SIZE = 64
STRIDE = 32
BATCH_SIZE = 4  # Adjust based on your GPU VRAM
MODEL_ID = "facebook/vjepa2-vitg-fpc64-384"
ROOT_DIR = "/ptmp/dduka/databases/ego4d/video_320px_15sec"


def process_video(video_path, model, processor, device):
    """Processes a single video in batches of sliding windows."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    hidden_size = model.config.hidden_size

    # Keep accumulators on CPU to save VRAM
    accumulated_features = torch.zeros((total_frames, hidden_size), dtype=torch.float32)
    counts = torch.zeros((total_frames, 1), dtype=torch.float32)

    batch_chunks = []
    batch_start_indices = []

    with torch.inference_mode():
        for start_idx in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            indices = np.arange(start_idx, end_idx)

            # Read frames and append to current batch list
            video_chunk = vr.get_batch(indices).asnumpy().transpose(0, 3, 1, 2)
            batch_chunks.append(video_chunk)
            batch_start_indices.append(start_idx)

            # Check if we hit the batch size OR if it's the very last window
            is_last_window = (start_idx + STRIDE) > (total_frames - WINDOW_SIZE)

            if len(batch_chunks) == BATCH_SIZE or is_last_window:
                # The processor can handle a list of 4D numpy arrays
                inputs = processor(batch_chunks, return_tensors="pt").to(
                    device, torch.bfloat16
                )

                outputs = model(**inputs)
                tokens = outputs.last_hidden_state

                current_b_size = len(batch_chunks)

                # Reshape with the dynamic batch size instead of hardcoding 1
                spatially_pooled = tokens.view(
                    current_b_size, 32, 24, 24, hidden_size
                ).mean(dim=(2, 3))

                # Interpolate
                window_features = torch.nn.functional.interpolate(
                    spatially_pooled.transpose(
                        1, 2
                    ),  # Swap to [B, C, T] for interpolate
                    size=WINDOW_SIZE,
                    mode="linear",
                    align_corners=False,
                ).transpose(
                    1, 2
                )  # Swap back to [B, T, C]

                # Move to CPU and accumulate
                window_features_cpu = window_features.cpu().to(torch.float32)

                for i, s_idx in enumerate(batch_start_indices):
                    e_idx = s_idx + WINDOW_SIZE
                    # If batch is 1, window_features_cpu might lose the batch dimension
                    # Ensure we are indexing correctly by checking dimensions
                    if current_b_size == 1 and window_features_cpu.dim() == 2:
                        accumulated_features[s_idx:e_idx] += window_features_cpu
                    else:
                        accumulated_features[s_idx:e_idx] += window_features_cpu[i]

                    counts[s_idx:e_idx] += 1

                # Clear batch lists for the next round
                batch_chunks = []
                batch_start_indices = []

    mask = counts > 0
    final_embeddings = torch.zeros_like(accumulated_features)
    final_embeddings[mask.squeeze()] = (
        accumulated_features[mask.squeeze()] / counts[mask.squeeze()]
    )

    return final_embeddings.numpy()


def merge_video_chunks(chunk_dir, temp_output_path):
    """Sorts and merges mp4 chunks using ffmpeg concat without re-encoding."""
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith(".mp4")]
    if not chunk_files:
        return False

    # Sort files numerically by their filename (e.g., '0.mp4', '15.mp4', '30.mp4')
    chunk_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Create a temporary text file listing the chunks for ffmpeg
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as list_file:
        for chunk in chunk_files:
            abs_chunk_path = os.path.join(chunk_dir, chunk)
            # ffmpeg format requires: file '/path/to/file.mp4'
            list_file.write(f"file '{abs_chunk_path}'\n")
        list_file_path = list_file.name

    # Run ffmpeg concat (-c copy ensures it's lightning fast and doesn't re-encode)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-f",
        "concat",  # Use concat demuxer
        "-safe",
        "0",  # Allow absolute paths in the text file
        "-i",
        list_file_path,
        "-c",
        "copy",  # Copy codecs stream directly
        temp_output_path,
    ]

    try:
        # capture_output suppresses ffmpeg's massive terminal output
        subprocess.run(cmd, check=True, capture_output=True)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Error merging videos in {chunk_dir}:\n{e.stderr.decode('utf-8')}")
        success = False
    finally:
        # Clean up the text file we gave to ffmpeg
        os.remove(list_file_path)

    return success


def main():
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    accelerator.print(f"Starting pipeline on {accelerator.num_processes} GPUs...")

    # 1. Load model and processor (fixed dtype deprecation warning)
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    # Accelerate handles moving the model to the correct GPU for this process
    model = accelerator.prepare(model)
    model.eval()

    # 2. Get video directories (only Main Process should scan the disk)
    all_video_dirs = []
    with accelerator.main_process_first():
        if os.path.exists(ROOT_DIR):
            # Find all directories ending in .mp4
            all_video_dirs = [
                os.path.join(ROOT_DIR, d)
                for d in os.listdir(ROOT_DIR)
                if d.endswith(".mp4") and os.path.isdir(os.path.join(ROOT_DIR, d))
            ]
        else:
            print(f"Warning: ROOT_DIR {ROOT_DIR} does not exist.")

    # 3. Define the rank-specific output file
    rank = accelerator.process_index
    temp_h5_file = (
        f"/ptmp/dduka/databases/ego4d/temp_embeddings_rank_{rank}.h5"
    )

    # 4. Use the context manager to split the directories, then process!
    with accelerator.split_between_processes(all_video_dirs) as process_video_dirs:
        accelerator.print(
            f"Rank {rank} processing {len(process_video_dirs)} directories..."
        )

        with h5py.File(temp_h5_file, "w") as f:
            # Disable tqdm on non-main processes to keep the terminal clean
            for vid_dir in tqdm(
                process_video_dirs, disable=not accelerator.is_main_process
            ):
                try:
                    # Extract ID from the directory name (e.g., "video_123.mp4" -> "video_123")
                    vid_id = os.path.basename(vid_dir).replace(".mp4", "")

                    # Define a safe temporary path for this specific rank to avoid process collisions
                    temp_merged_video = os.path.join(
                        tempfile.gettempdir(), f"merged_{vid_id}_rank{rank}.mp4"
                    )

                    # Merge chunks into one file temporarily
                    if merge_video_chunks(vid_dir, temp_merged_video):

                        # Extract features
                        features = process_video(
                            temp_merged_video, model, processor, device
                        )

                        f.create_dataset(vid_id, data=features, compression="gzip")

                        if os.path.exists(temp_merged_video):
                            os.remove(temp_merged_video)
                    else:
                        print(f"Rank {rank}: No valid chunks found in {vid_dir}")

                except Exception as e:
                    print(f"Rank {rank} failed on {vid_dir}: {e}")

    # 5. Wait for all GPUs to finish their individual files
    accelerator.wait_for_everyone()

    # 6. Merge the temporary files into one master file (Main Process only)
    if accelerator.is_main_process:
        master_file = (
            "/ptmp/dduka/databases/ego4d/vjepa_embeddings_master.h5"
        )
        print(f"All GPUs finished. Merging into {master_file}...")

        with h5py.File(master_file, "w") as f_out:
            for i in range(accelerator.num_processes):
                temp_file = f"/ptmp/dduka/databases/ego4d/temp_embeddings_rank_{i}.h5"
                if os.path.exists(temp_file):
                    with h5py.File(temp_file, "r") as f_in:
                        for key in f_in.keys():
                            f_in.copy(key, f_out)
                    os.remove(temp_file)

        print("Merge complete! Pipeline finished successfully.")


if __name__ == "__main__":
    main()
