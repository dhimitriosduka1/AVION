import torch
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt

VIDEO_ID = "401c588f-907f-4559-b6f7-8180ae6969ec"
VIDEO_FEATURES_PATH = f"/dais/fs/scratch/dduka/databases/ego4d/extracted_features/{VIDEO_ID}.pt"
DATASET_PATH = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
FPS = 7.5

def construct_mog_distribution(timestamps, std, num_points=1000):
    if not timestamps:
        return np.array([]), np.array([])
        
    x = np.linspace(min(timestamps) - 3 * std, max(timestamps) + 3 * std, num_points)
    y = np.zeros_like(x)
    
    for t in timestamps:
        y += (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - t) / std) ** 2)
        
    y /= len(timestamps)
    return x, y

# --- Feature Loading & Cosine Similarity ---
features = torch.load(VIDEO_FEATURES_PATH)
print(f"Loaded features for video ID {VIDEO_ID} with shape {features.shape}.")

features_normalized = F.normalize(features, p=2, dim=1)
cos_sim = 1.0 - (features_normalized[:-1] * features_normalized[1:]).sum(dim=1).cpu().numpy()

# Calculate Running Average (Smoothing)
# 15 frames at 7.5 FPS = 2-second moving average window
window_size = 15  
kernel = np.ones(window_size) / window_size
# mode='same' ensures the smoothed array stays the exact same length as the original
cos_sim_smoothed = np.convolve(cos_sim, kernel, mode='same')

# --- Dataset Loading & MoG ---
with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

filtered_data = [item for item in data if item[1] == VIDEO_ID]
print(f"Found {len(filtered_data)} entries for video ID {VIDEO_ID}.")
filtered_data.sort(key=lambda x: x[2])

timestamps = [item[2] for item in filtered_data]
std_dev = 2.0 
x_axis, mog_y = construct_mog_distribution(timestamps, std_dev)

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top Subplot: Mixture of Gaussians
if len(x_axis) > 0:
    axes[0].plot(x_axis, mog_y, label=f'MoG (std={std_dev})', color='blue', linewidth=2)
    axes[0].plot(timestamps, np.zeros_like(timestamps), '|', color='red', markersize=15, label='Actual Timestamps')
    axes[0].set_title(f'Mixture of Gaussians for Ego4D Video: {VIDEO_ID}')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, "No timestamps found", ha='center', va='center')

# Bottom Subplot: Consecutive Frame Cosine Similarity
time_x = np.arange(len(cos_sim)) / FPS

# Plot raw data faintly in the background
axes[1].plot(time_x, cos_sim, color='green', alpha=0.3, linewidth=1.0, label='Raw Cosine Sim')

# Plot the running average boldly on top
axes[1].plot(time_x, cos_sim_smoothed, color='darkgreen', linewidth=2.0, label=f'Running Avg (Window={window_size})')

axes[1].set_title('Consecutive Frame Similarity (Smoothed)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Cosine Similarity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_path = f"{VIDEO_ID}_analysis_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Successfully saved the combined plot to {save_path}")

plt.close()