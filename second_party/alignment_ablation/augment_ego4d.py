import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. AUGMENTATION GENERATOR (.pkl)
# ==========================================

def apply_safe_dynamic_augmentations_pkl(input_pkl, output_dir):
    print(f"Loading dataset from {input_pkl}...")
    
    # 1. Load the pickle file (List of tuples/lists)
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
        
    # Load into Pandas using integer column indices: 
    # 0: video_id, 1: start_second, 2: end_second, 3: caption
    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Base Metrics using numeric indices
    df['duration'] = df[2] - df[1]
    
    # Extract Video Ceilings (The max known time for each video using index 0 and 2)
    df['video_max_sec'] = df.groupby(0)[2].transform('max')

    # Compute Dataset Statistics
    min_dur = df['duration'].min()
    max_dur = df['duration'].max()
    mean_dur = df['duration'].mean()

    print(f"Dataset Duration Stats: min={min_dur:.2f}s, mean={mean_dur:.2f}s, max={max_dur:.2f}s")
    
    # Define Dynamic Experiments
    # Format: "experiment_name": ("operation_type", value)
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
            # Add n/2 to the end, subtract n/2 from the start
            df_exp['new_start_sec'] = df_exp[1] - (n / 2.0)
            df_exp['new_stop_sec'] = df_exp[2] + (n / 2.0)
            
        elif op_type == "scale":
            N = val
            # Scale duration by N, extending symmetrically from the center
            extra_time = df_exp['duration'] * (N - 1.0)
            df_exp['new_start_sec'] = df_exp[1] - (extra_time / 2.0)
            df_exp['new_stop_sec'] = df_exp[2] + (extra_time / 2.0)

        # --- THE SAFETY BOUNDARIES ---
        df_exp['new_start_sec'] = df_exp['new_start_sec'].clip(lower=0.0)
        df_exp['new_start_sec'] = np.minimum(df_exp['new_start_sec'], df_exp['video_max_sec'] - 0.1)
        df_exp['new_stop_sec'] = np.minimum(df_exp['new_stop_sec'], df_exp['video_max_sec'])
        df_exp['new_stop_sec'] = np.maximum(df_exp['new_stop_sec'], df_exp['new_start_sec'] + 0.1)

        # Map back to the original index columns
        df_exp[1] = df_exp['new_start_sec']
        df_exp[2] = df_exp['new_stop_sec']

        # Extract only columns 0, 1, 2, 3 and convert back to a list of tuples
        final_data = list(df_exp[[0, 1, 2, 3]].itertuples(index=False, name=None))
        
        # Dump to new .pkl file
        output_path = os.path.join(output_dir, f"ego4d_{exp_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f)
        processed_names.append(exp_name)

    print(f"All splits saved successfully to {output_dir}\n")
    return processed_names

# ==========================================
# 2. DISTRIBUTION PLOTTER (.pkl)
# ==========================================

def plot_experiment_distributions_pkl(original_pkl, augmented_dir, experiment_names):
    print("Generating distribution plots...")
    
    # Load Original Data
    with open(original_pkl, 'rb') as f:
        orig_data = pickle.load(f)
        
    df_orig = pd.DataFrame(orig_data)
    df_orig['duration'] = df_orig[2] - df_orig[1]
    
    max_plot_val = df_orig['duration'].quantile(0.99)
    
    durations_dict = {"Original GT": df_orig['duration']}
    
    # Dynamically load the generated experiments
    for exp_name in experiment_names:
        file_path = os.path.join(augmented_dir, f"ego4d_{exp_name}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                exp_data = pickle.load(f)
            df_exp = pd.DataFrame(exp_data)
            dur = df_exp[2] - df_exp[1]
            durations_dict[exp_name] = dur
        else:
            print(f"Warning: Could not find {file_path}")

    # Setup Grid dynamically based on number of experiments
    num_plots = len(durations_dict)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))
    
    # Flatten axes for easy iteration, handling cases where there's only one row
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
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
    plt.suptitle("Action Duration Distributions (Full pkl preserved)", fontsize=16, fontweight='bold')

    plot_path = os.path.join(augmented_dir, "distribution_grid.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    # Define your paths for the .pkl files
    INPUT_PKL = '/ptmp/dduka/databases/ego4d/augemented_gt_labels/gt_train.pkl'
    OUTPUT_DIR = '/ptmp/dduka/databases/ego4d/augemented_gt_labels/'

    # Run the pipeline and capture the generated experiment names
    ran_experiments = apply_safe_dynamic_augmentations_pkl(INPUT_PKL, OUTPUT_DIR)
    
    # Plot using the captured experiment names
    plot_experiment_distributions_pkl(INPUT_PKL, OUTPUT_DIR, ran_experiments)