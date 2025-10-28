# In check_merged_data.py

import numpy as np
import os

def check_npz_file(file_path, expected_key):
    """
    Checks the structure and content of a merged dataset .npz file,
    performing a comprehensive shape check on ALL sequences for ALL subjects.
    """
    print(f"\n--- Checking File: {file_path} ---")
    
    # Determine expected coordinate dimension based on the key
    if expected_key == 'positions_3d':
        expected_coord_dim = 3
        data_type = '3D'
    elif expected_key == 'positions_2d':
        expected_coord_dim = 2
        data_type = '2D'
    else:
        print(f"❌ ERROR: Unknown expected key '{expected_key}'. Cannot determine expected shape.")
        return

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
        
        # Separate subjects
        h36m_subjects = sorted([s for s in subjects if not s.endswith('_MPI')])
        mpi_subjects = sorted([s for s in subjects if s.endswith('_MPI')])
        
        print(f" -> Found {len(h36m_subjects)} Human36M subjects.")
        print(f" -> Found {len(mpi_subjects)} MPI-INF-3DHP subjects.")

        # Initialize counters for comprehensive check
        total_sequences = 0
        sequences_with_errors = 0
        
        # --- Comprehensive Check Function ---
        def check_sequence_shape(sequence, subject, action, seq_idx):
            nonlocal total_sequences, sequences_with_errors
            total_sequences += 1
            
            is_valid = True
            error_msg = ""
            
            if not isinstance(sequence, np.ndarray):
                is_valid = False
                error_msg += "Not a numpy array. "
            
            elif len(sequence.shape) != 3:
                is_valid = False
                error_msg += f"Expected 3 dims, got {len(sequence.shape)}. "
                
            elif sequence.shape[1] != 17:
                is_valid = False
                error_msg += f"Joints: Expected 17, got {sequence.shape[1]}. "
                
            elif sequence.shape[2] != expected_coord_dim:
                is_valid = False
                error_msg += f"Coords: Expected {expected_coord_dim} for {data_type}, got {sequence.shape[2]}. "

            if not is_valid:
                sequences_with_errors += 1
                print(f"❌ ERROR in {subject}/{action}[{seq_idx}] (Shape: {sequence.shape}): {error_msg}")
            
            return is_valid
        
        # --- Iterate over all data ---
        print(f"\n--- Running Comprehensive Shape Check for all sequences (Expected Shape: (F, 17, {expected_coord_dim})) ---")
        
        for subject, subject_data in main_dict.items():
            for action, sequences in subject_data.items():
                for seq_idx, sequence in enumerate(sequences):
                    # Check each sequence in the list
                    check_sequence_shape(sequence, subject, action, seq_idx)

        # --- Final Summary ---
        print("\n" + "="*50)
        print(f"FINAL CHECK SUMMARY FOR {expected_key}:")
        print(f"Total Sequences Checked: {total_sequences}")
        
        if sequences_with_errors == 0 and total_sequences > 0:
            print(f"✅ SUCCESS: All {total_sequences} sequences passed the shape check!")
        elif sequences_with_errors > 0:
            print(f"❌ FAILURE: {sequences_with_errors} sequences out of {total_sequences} had shape errors.")
        else:
            print("⚠️ WARNING: No sequences were found to check (Data dictionary might be empty).")
        
        print("="*50)


    except Exception as e:
        print(f"❌ An unexpected error occurred while checking the file: {e}")

if __name__ == '__main__':
    # Ensure you have 'data/data_3d_merged.npz' and 'data/data_2d_merged.npz' created
    merged_3d_path = 'data/data_3d_merged.npz'
    merged_2d_path = 'data/data_2d_merged.npz'
    
    check_npz_file(merged_3d_path, 'positions_3d')
    print("\n" + "="*50 + "\n") # Separator for clarity
    check_npz_file(merged_2d_path, 'positions_2d')