import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def time_to_sec(time_str):
    """Converts HH:MM:SS.ms string to float seconds."""
    h, m, s = str(time_str).split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def sec_to_time(seconds):
    """Converts float seconds back to HH:MM:SS.ms format."""
    seconds = max(0, seconds)  
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

# ==========================================
# 2. AUGMENTATION GENERATOR
# ==========================================

def apply_safe_dynamic_augmentations(input_csv, output_dir):
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Base Metrics
    df['start_sec'] = df['start_timestamp'].apply(time_to_sec)
    df['stop_sec'] = df['stop_timestamp'].apply(time_to_sec)
    df['duration'] = df['stop_sec'] - df['start_sec']
    
    safe_duration = df['duration'].replace(0, 0.001)
    df['fps'] = (df['stop_frame'] - df['start_frame']) / safe_duration
    
    # Extract Video Ceilings (The max known time/frame for each video)
    df['video_max_sec'] = df.groupby('video_id')['stop_sec'].transform('max')
    df['video_max_frame'] = df.groupby('video_id')['stop_frame'].transform('max')

    # Compute Dataset Statistics
    min_dur = df['duration'].min()
    max_dur = df['duration'].max()
    mean_dur = df['duration'].mean()

    print(f"Dataset Duration Stats: min={min_dur:.2f}s, mean={mean_dur:.2f}s, max={max_dur:.2f}s")
    
    # Define Dynamic Experiments
    experiments = {
        "01_expand_by_min": (-min_dur, min_dur),
        "02_expand_by_mean_10pct": (-(mean_dur * 0.1), (mean_dur * 0.1)),
        "03_shift_forward_mean": (mean_dur, mean_dur),
        "04_shift_backward_mean": (-mean_dur, -mean_dur),
        "05_contract_by_min": (min_dur, -min_dur),
        "06_expand_by_mean_20pct": (-(mean_dur * 0.2), (mean_dur * 0.2)),
        "07_contract_by_mean_20pct": ((mean_dur * 0.2), -(mean_dur * 0.2)),
    }

    # Process Each Experiment
    for exp_name, (start_mod, stop_mod) in experiments.items():
        print(f"  -> Processing: {exp_name}...")
        df_exp = df.copy()
        
        df_exp['new_start_sec'] = df_exp['start_sec'] + start_mod
        df_exp['new_stop_sec'] = df_exp['stop_sec'] + stop_mod
        
        # 1-Second Contraction Rule
        if "contract" in exp_name:
            df_exp['new_dur'] = df_exp['new_stop_sec'] - df_exp['new_start_sec']
            mask_needs_fix = (df_exp['new_dur'] < 1.0) & (df_exp['duration'] >= 1.0)
            midpoint = (df_exp['start_sec'] + df_exp['stop_sec']) / 2.0
            
            df_exp.loc[mask_needs_fix, 'new_start_sec'] = midpoint[mask_needs_fix] - 0.5
            df_exp.loc[mask_needs_fix, 'new_stop_sec'] = midpoint[mask_needs_fix] + 0.5
            
            mask_too_short = df_exp['duration'] < 1.0
            df_exp.loc[mask_too_short, 'new_start_sec'] = df_exp.loc[mask_too_short, 'start_sec']
            df_exp.loc[mask_too_short, 'new_stop_sec'] = df_exp.loc[mask_too_short, 'stop_sec']

        # --- THE SAFETY BOUNDARIES ---
        # Floor: Cannot drop below 0 seconds
        df_exp['new_start_sec'] = df_exp['new_start_sec'].clip(lower=0.0)
        
        # Ceiling: Cannot exceed the maximum known time of the video
        df_exp['new_stop_sec'] = np.minimum(df_exp['new_stop_sec'], df_exp['video_max_sec'])
        
        # Logic Check: Stop time must still be greater than start time
        df_exp['new_stop_sec'] = np.maximum(df_exp['new_stop_sec'], df_exp['new_start_sec'] + 0.1)

        # Update Frames via FPS
        start_frame_delta = (df_exp['new_start_sec'] - df_exp['start_sec']) * df_exp['fps']
        stop_frame_delta = (df_exp['new_stop_sec'] - df_exp['stop_sec']) * df_exp['fps']
        
        df_exp['start_frame'] = (df_exp['start_frame'] + start_frame_delta).round().astype(int)
        df_exp['stop_frame'] = (df_exp['stop_frame'] + stop_frame_delta).round().astype(int)
        
        # Frame Safety Checks (Floor and Ceiling)
        df_exp['start_frame'] = df_exp['start_frame'].clip(lower=0)
        df_exp['stop_frame'] = np.minimum(df_exp['stop_frame'], df_exp['video_max_frame'])
        df_exp['stop_frame'] = np.maximum(df_exp['stop_frame'], df_exp['start_frame'] + 1)

        # Format back to strings
        df_exp['start_timestamp'] = df_exp['new_start_sec'].apply(sec_to_time)
        df_exp['stop_timestamp'] = df_exp['new_stop_sec'].apply(sec_to_time)

        # Clean up
        cols_to_drop = ['start_sec', 'stop_sec', 'duration', 'fps', 'new_start_sec', 'new_stop_sec', 'video_max_sec', 'video_max_frame']
        if 'new_dur' in df_exp.columns:
            cols_to_drop.append('new_dur')
            
        df_exp = df_exp.drop(columns=cols_to_drop)
        output_path = os.path.join(output_dir, f"ek100_{exp_name}.csv")
        df_exp.to_csv(output_path, index=False)
        
    print(f"All splits saved successfully to {output_dir}\n")

# ==========================================
# 3. DISTRIBUTION PLOTTER
# ==========================================

def plot_experiment_distributions(original_csv, augmented_dir):
    print("Generating distribution plots...")
    
    # Load Original Data
    df_orig = pd.read_csv(original_csv)
    df_orig['duration'] = df_orig['stop_timestamp'].apply(time_to_sec) - df_orig['start_timestamp'].apply(time_to_sec)
    max_plot_val = df_orig['duration'].quantile(0.99)
    
    experiment_names = [
        "01_expand_by_min",
        "02_expand_by_mean_10pct",
        "03_shift_forward_mean",
        "04_shift_backward_mean",
        "05_contract_by_min",
        "06_expand_by_mean_20pct",
        "07_contract_by_mean_20pct"
    ]
    
    durations_dict = {"Original GT": df_orig['duration']}
    
    for exp_name in experiment_names:
        file_path = os.path.join(augmented_dir, f"ek100_{exp_name}.csv")
        if os.path.exists(file_path):
            df_exp = pd.read_csv(file_path)
            dur = df_exp['stop_timestamp'].apply(time_to_sec) - df_exp['start_timestamp'].apply(time_to_sec)
            durations_dict[exp_name] = dur
        else:
            print(f"Warning: Could not find {file_path}")

    # Setup Grid
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Plot each distribution
    for i, (title, durations) in enumerate(durations_dict.items()):
        ax = axes[i]
        
        ax.hist(durations.clip(upper=max_plot_val), bins=50, color='royalblue', edgecolor='black', alpha=0.7)
        
        mean_val = durations.mean()
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}s')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Duration (Seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for j in range(len(durations_dict), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("Action Duration Distributions Across Ablation Splits", fontsize=16, y=1.02, fontweight='bold')
    
    plot_path = os.path.join(augmented_dir, "distribution_grid.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {plot_path}")
    
    plt.show()

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # Define your specific paths
    INPUT_CSV = '/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv'
    OUTPUT_DIR = '/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/augmented_cls/'

    # Run the pipeline
    apply_safe_dynamic_augmentations(INPUT_CSV, OUTPUT_DIR)
    plot_experiment_distributions(INPUT_CSV, OUTPUT_DIR)