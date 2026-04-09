import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def time_to_sec(time_str):
    """Converts HH:MM:SS.ms string to float seconds."""
    try:
        h, m, s = str(time_str).split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return 0.0

def sec_to_time(seconds):
    """Converts float seconds back to HH:MM:SS.ms format."""
    seconds = max(0, seconds)  
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

# ==========================================
# 2. AUGMENTATION GENERATOR (PRESERVING ALL COLUMNS)
# ==========================================

def apply_safe_dynamic_augmentations(input_csv, output_dir):
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Base Metrics for the math logic
    df['_start_sec'] = df['start_timestamp'].apply(time_to_sec)
    df['_stop_sec'] = df['stop_timestamp'].apply(time_to_sec)
    df['_duration'] = df['_stop_sec'] - df['_start_sec']
    
    # Extract Video Ceilings
    df['_video_max_sec'] = df.groupby('video_id')['_stop_sec'].transform('max')

    # Define Ablation Experiments
    experiments = {
        # Method 1: Add n seconds total (subtract n/2 from start, add n/2 to end)
        "add_1_sec": ("add", 1.0),
        "add_2_sec": ("add", 2.0),
        "add_3_sec": ("add", 3.0),
        "add_4_sec": ("add", 4.0),
        "add_5_sec": ("add", 5.0),
        "add_6_sec": ("add", 6.0),
        "add_7_sec": ("add", 7.0),
        "add_8_sec": ("add", 8.0),
        # Method 2: Scale segment by factor of N
        "scale_1_1x": ("scale", 1.1),
        "scale_1_2x": ("scale", 1.2),
        "scale_1_3x": ("scale", 1.3),
        "scale_1_4x": ("scale", 1.4),
        "scale_1_5x": ("scale", 1.5),
        "scale_1_6x": ("scale", 1.6),
        "scale_1_7x": ("scale", 1.7),
        "scale_1_8x": ("scale", 1.8),
        "scale_1_9x": ("scale", 1.9),
        "scale_2_0x": ("scale", 2.0),
        "scale_2_1x": ("scale", 2.1),
        "scale_2_2x": ("scale", 2.2),
        "scale_2_3x": ("scale", 2.3),
        "scale_2_4x": ("scale", 2.4),
        "scale_2_5x": ("scale", 2.5)
    }

    processed_names = []

    for exp_name, (op_type, val) in experiments.items():
        print(f"  -> Processing: {exp_name}...")
        df_exp = df.copy()
        
        if op_type == "add":
            n = val
            df_exp['_new_start_sec'] = df_exp['_start_sec'] - (n / 2.0)
            df_exp['_new_stop_sec'] = df_exp['_stop_sec'] + (n / 2.0)
            
        elif op_type == "scale":
            N = val
            extra_time = df_exp['_duration'] * (N - 1.0)
            df_exp['_new_start_sec'] = df_exp['_start_sec'] - (extra_time / 2.0)
            df_exp['_new_stop_sec'] = df_exp['_stop_sec'] + (extra_time / 2.0)

        # --- SAFETY BOUNDARIES ---
        df_exp['_new_start_sec'] = df_exp['_new_start_sec'].clip(lower=0.0)
        df_exp['_new_start_sec'] = np.minimum(df_exp['_new_start_sec'], df_exp['_video_max_sec'] - 0.1)
        df_exp['_new_stop_sec'] = np.minimum(df_exp['_new_stop_sec'], df_exp['_video_max_sec'])
        df_exp['_new_stop_sec'] = np.maximum(df_exp['_new_stop_sec'], df_exp['_new_start_sec'] + 0.1)

        # 1. Update the required timestamp columns
        df_exp['start_timestamp'] = df_exp['_new_start_sec'].apply(sec_to_time)
        df_exp['stop_timestamp'] = df_exp['_new_stop_sec'].apply(sec_to_time)

        # 2. Set frames to None as requested
        df_exp['start_frame'] = None
        df_exp['stop_frame'] = None

        # 3. Drop only the temporary calculation columns (prefixed with underscore)
        temp_cols = [c for c in df_exp.columns if c.startswith('_')]
        df_exp = df_exp.drop(columns=temp_cols)
        
        output_path = os.path.join(output_dir, f"ek100_{exp_name}.csv")
        df_exp.to_csv(output_path, index=False)
        processed_names.append(exp_name)
        
    print(f"All splits saved successfully to {output_dir}\n")
    return processed_names

# ==========================================
# 3. DISTRIBUTION PLOTTER
# ==========================================

def plot_experiment_distributions(original_csv, augmented_dir, experiment_names):
    print("Generating distribution plots...")
    
    df_orig = pd.read_csv(original_csv)
    df_orig['duration'] = df_orig['stop_timestamp'].apply(time_to_sec) - df_orig['start_timestamp'].apply(time_to_sec)
    
    max_plot_val = df_orig['duration'].quantile(0.99)
    durations_dict = {"Original GT": df_orig['duration']}
    
    for exp_name in experiment_names:
        file_path = os.path.join(augmented_dir, f"ek100_{exp_name}.csv")
        if os.path.exists(file_path):
            df_exp = pd.read_csv(file_path)
            dur = df_exp['stop_timestamp'].apply(time_to_sec) - df_exp['start_timestamp'].apply(time_to_sec)
            durations_dict[exp_name] = dur

    num_plots = len(durations_dict)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i, (title, durations) in enumerate(durations_dict.items()):
        ax = axes[i]
        ax.hist(durations.clip(upper=max_plot_val), bins=50, color='darkorange', edgecolor='black', alpha=0.7)
        mean_val = durations.mean()
        ax.axvline(mean_val, color='blue', linestyle='dashed', label=f'Mean: {mean_val:.2f}s')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Seconds')
        ax.legend()

    for j in range(len(durations_dict), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Action Duration Distributions (Full CSV preserved)", fontsize=16, fontweight='bold')
    
    plot_path = os.path.join(augmented_dir, "distribution_grid.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    INPUT_CSV = '/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv'
    OUTPUT_DIR = '/ptmp/dduka/databases/EK100/epic-kitchens-100-annotations/retrieval_annotations/augmented_mir/'

    ran_experiments = apply_safe_dynamic_augmentations(INPUT_CSV, OUTPUT_DIR)
    plot_experiment_distributions(INPUT_CSV, OUTPUT_DIR, ran_experiments)