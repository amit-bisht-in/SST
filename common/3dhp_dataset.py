# In convert_3dhp.py

import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm

# 1. Define the root path to your downloaded dataset
dataset_root = r'data\mpi_inf_3dhp'

# Dictionaries to hold all the data from all subjects and sequences
# The structure will be {'S1_Seq1': pose_data, 'S1_Seq2': pose_data, ...}
all_poses_3d = {}
all_poses_2d = {}

# 2. Get a list of all subject folders (e.g., 'S1', 'S2', ...)

subject_folders = ['S8']

print(f"Found {len(subject_folders)} subjects: {subject_folders}")

# 3. Loop through each subject
for subject_name in tqdm(subject_folders, desc="Processing Subjects"):
    subject_path = os.path.join(dataset_root, subject_name)
    
    # 4. Get a list of all sequence folders for that subject
    sequence_folders = sorted([d for d in os.listdir(subject_path) if d.startswith('Seq') and os.path.isdir(os.path.join(subject_path, d))])
    
    # 5. Loop through each sequence
    for seq_name in sequence_folders:
        annot_path = os.path.join(subject_path, seq_name, 'annot.mat')
        
        if os.path.exists(annot_path):
            # 6. Load the data for this specific sequence
            annotations = sio.loadmat(annot_path)
            
            # 7. Reshape the flattened data
            num_joints = 28
            poses_3d_flat = annotations['annot3'][0][0]
            poses_3d_reshaped = poses_3d_flat.reshape(-1, num_joints, 3)
            
            poses_2d_flat = annotations['annot2'][0][0]
            poses_2d_reshaped = poses_2d_flat.reshape(-1, num_joints, 2)
            
            # 8. Store the data under a unique key (e.g., "S1_Seq1")
            action_name = f"{subject_name}_{seq_name}"
            all_poses_3d[action_name] = poses_3d_reshaped
            all_poses_2d[action_name] = poses_2d_reshaped

print("\nProcessing complete.")
print(f"Gathered data for {len(all_poses_3d)} total video sequences.")
print(f"Example 3D shape for one sequence: {list(all_poses_3d.values())[0].shape}")

# --- YOUR NEXT STEPS WOULD GO HERE ---
# 9. Write a function to convert the 28-joint skeleton to the 17-joint H36M skeleton.
# 10. Loop through the `all_poses_3d` and `all_poses_2d` dictionaries, apply the conversion.
# 11. Save the final, converted data into new .npz files.
# ------------------------------------