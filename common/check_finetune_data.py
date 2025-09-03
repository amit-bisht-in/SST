# In check_merged_data.py

import numpy as np
import os

def check_npz_file(file_path, expected_key):
    """
    Checks the structure and content of a merged dataset .npz file.
    """
    print(f"\n--- Checking File: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found.")
        return

    try:
        data = np.load(file_path, allow_pickle=True)
        
        if expected_key not in data:
            print(f"❌ ERROR: Expected key '{expected_key}' not found in the file.")
            return

        main_dict = data[expected_key].item()
        subjects = list(main_dict.keys())
        print(f"✅ File loaded successfully. Found {len(subjects)} total subjects.")
        
        # Check if both original and new subjects are present
        h36m_subjects = [s for s in subjects if not s.endswith('_MPI')]
        mpi_subjects = [s for s in subjects if s.endswith('_MPI')]
        
        print(f" -> Found {len(h36m_subjects)} Human36M subjects (e.g., {h36m_subjects[0]})")
        print(f" -> Found {len(mpi_subjects)} MPI-INF-3DHP subjects (e.g., {mpi_subjects[0]})")

        # Check the shape of a sample sequence from the newly converted data
        sample_mpi_subject = mpi_subjects[0]
        first_action = list(main_dict[sample_mpi_subject].keys())[0]
        first_sequence = main_dict[sample_mpi_subject][first_action][0]
        
        print(f"\nChecking a sample from the new data ({sample_mpi_subject}/{first_action})...")
        print(f"Shape of the sample sequence: {first_sequence.shape}")
        
        num_joints = first_sequence.shape[1]
        if num_joints == 17:
            print("✅ The joint count is 17, which is correct for the H36M format.")
        else:
            print(f"❌ WARNING: The joint count is {num_joints}, but it should be 17.")

    except Exception as e:
        print(f"❌ An error occurred while checking the file: {e}")

if __name__ == '__main__':
    merged_3d_path = 'data/data_3d_merged.npz'
    merged_2d_path = 'data/data_2d_merged.npz'
    
    check_npz_file(merged_3d_path, 'positions_3d')
    check_npz_file(merged_2d_path, 'positions_2d')