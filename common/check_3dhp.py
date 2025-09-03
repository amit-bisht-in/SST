# In your inspector script (e.g., 3dhp_dataset.py)

import scipy.io as sio
import numpy as np
import os

# --- IMPORTANT: SET THE FULL, CORRECT PATH ---
annot_path = r'D:\Work\ML\SST\data\mpi_inf_3dhp\S5\Seq1\annot.mat'
# ---------------------------------------------------

print(f"--- Inspecting file: {annot_path} ---")

if not os.path.exists(annot_path):
    print(f"ERROR: File not found. Please update the 'annot_path' variable.")
else:
    try:
        # Use scipy.io.loadmat for older MAT-file formats
        annotations = sio.loadmat(annot_path)
        
        print("\nFile loaded successfully with SciPy. Here are the keys:")
        print(list(annotations.keys()))
        
        print("\n--- Exploring each key's shape ---")
        for key, value in annotations.items():
            if key.startswith('__'):
                continue
            
            print(f"\n--> Key: '{key}'")
            
            if isinstance(value, np.ndarray):
                print(f"    Shape: {value.shape}")
                print(f"    Data Type: {value.dtype}")
            else:
                print(f"    Type: {type(value)}")

    except Exception as e:
        print(f"An error occurred: {e}")

    
print("\n--- Inspecting Nested Array Shapes ---")

# Check the 3D data's nested shape
if 'annot3' in annotations:
    try:
        # Access the data for the first camera (index 0)
        pose_data_3d = annotations['annot3'][0][0] 
        print(f"\nShape of 3D data for one camera (from 'annot3'): {pose_data_3d.shape}")
        print("(This likely corresponds to (num_joints, 3_coordinates, num_frames))")
    except IndexError:
        print("\nCould not access nested 3D data. 'annot3' might be empty.")

# Check the 2D data's nested shape
if 'annot2' in annotations:
    try:
        # Access the data for the first camera (index 0)
        pose_data_2d = annotations['annot2'][0][0]
        print(f"\nShape of 2D data for one camera (from 'annot2'): {pose_data_2d.shape}")
        print("(This likely corresponds to (num_joints, 2_coordinates, num_frames))")
    except IndexError:
        print("\nCould not access nested 2D data. 'annot2' might be empty.")