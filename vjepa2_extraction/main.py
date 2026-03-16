import os
import torch
import numpy as np
import glob
import subprocess
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from tqdm import tqdm
import torch.multiprocessing as mp

# --- Configuration ---
VIDEO_DIR = "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/"
OUTPUT_DIR = "/dais/fs/scratch/dduka/databases/ego4d/extracted_features/"
TEMP_MERGE_DIR = "/dais/fs/scratch/dduka/temp_merged_videos/"
HF_REPO = "facebook/vjepa2-vitg-fpc64-384"
WINDOW_FRAMES = 64
WINDOW_STRIDE = 8
SAMPLE_RATE = 4
BATCH_SIZE = 16 

def merge_chunks(video_id_dir):
    chunks = sorted(glob.glob(os.path.join(video_id_dir, "*.mp4")), key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    if not chunks: return None

    video_id = os.path.basename(video_id_dir).replace(".mp4", "")
    merged_out = os.path.join(TEMP_MERGE_DIR, f"{video_id}_merged.mp4")
    
    list_file = merged_out + ".txt"
    with open(list_file, "w") as f:
        for c in chunks:
            f.write(f"file '{os.path.abspath(c)}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file, "-c", "copy", merged_out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(list_file)
    
    return merged_out

def spatial_average_pool(token_embeddings, window_frames):
    B, seq_len, hidden_size = token_embeddings.shape
    num_temporal_steps = window_frames // 2
    num_spatial_patches = seq_len // num_temporal_steps
    grid = token_embeddings.view(B, num_temporal_steps, num_spatial_patches, hidden_size)
    tubelet_embeddings = grid.mean(dim=2)
    return tubelet_embeddings.repeat_interleave(2, dim=1)

def worker(rank, video_queue):
    device = torch.device(f"cuda:{rank}")
    model = AutoModel.from_pretrained(HF_REPO, torch_dtype=torch.float16).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)

    while True:
        video_dir_path = video_queue.get()
        if video_dir_path is None: 
            video_queue.task_done()
            break 
        
        video_id = os.path.basename(video_dir_path).replace(".mp4", "")
        save_path = os.path.join(OUTPUT_DIR, f"{video_id}.pt")
        
        if os.path.exists(save_path):
            video_queue.task_done()
            continue

        merged_video = None
        try:
            # Step 1: Merge the chunks
            merged_video = merge_chunks(video_dir_path)
            if merged_video is None: raise ValueError("No chunks found")

            # Step 2: Extract with torchcodec
            vr = VideoDecoder(merged_video)
            indices = np.arange(0, vr.num_frames, SAMPLE_RATE)
            sampled = vr.get_frames_at(indices=indices).data
            num_sampled = len(sampled)

            if num_sampled >= WINDOW_FRAMES:
                accum = torch.zeros(num_sampled, model.config.hidden_size)
                counts = torch.zeros(num_sampled, 1)

                starts = list(range(0, num_sampled - WINDOW_FRAMES + 1, WINDOW_STRIDE))
                if (num_sampled - WINDOW_FRAMES) % WINDOW_STRIDE != 0:
                    starts.append(num_sampled - WINDOW_FRAMES)

                for i in range(0, len(starts), BATCH_SIZE):
                    batch_starts = starts[i : i + BATCH_SIZE]
                    batch_windows = [sampled[s : s + WINDOW_FRAMES] for s in batch_starts]
                    inputs = processor(batch_windows, return_tensors="pt").to(device)
                    inputs = {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        tokens = model.get_vision_features(**inputs)
                    
                    per_frame_batch = spatial_average_pool(tokens, WINDOW_FRAMES).to(torch.float32).cpu()
                    for j, start in enumerate(batch_starts):
                        accum[start : start + WINDOW_FRAMES] += per_frame_batch[j]
                        counts[start : start + WINDOW_FRAMES] += 1

                torch.save(accum / counts, save_path)
            
        except Exception as e:
            print(f"Error on GPU {rank} | {video_id}: {e}")
        finally:
            if merged_video and os.path.exists(merged_video):
                os.remove(merged_video)
        
        video_queue.task_done()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_MERGE_DIR, exist_ok=True)
    
    video_id_dirs = [d for d in glob.glob(os.path.join(VIDEO_DIR, "*")) if os.path.isdir(d)]
    num_gpus = torch.cuda.device_count()
    
    queue = mp.JoinableQueue()
    for d in video_id_dirs: queue.put(d)
    for _ in range(num_gpus): queue.put(None)

    print(f"Processing {len(video_id_dirs)} Ego4D videos on {num_gpus} GPUs... 🦍")
    processes = [mp.Process(target=worker, args=(i, queue)) for i in range(num_gpus)]
    for p in processes: p.start()
    
    queue.join()
    for p in processes: p.join()
    print("Done!")