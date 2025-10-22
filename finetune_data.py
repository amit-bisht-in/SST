# In prepare_finetune_dataset.py

import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path to allow importing from `common`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.camera import normalize_screen_coordinates

def mpi_to_h36m_skeleton(poses):
    """
    Converts a 28-joint MPI-INF-3DHP pose sequence to a 17-joint H36M pose sequence.
    """
    h36m_poses = np.zeros((poses.shape[0], 17, poses.shape[2]), dtype=np.float32)
    # ... (Mapping logic remains the same) ...
    return h36m_poses

# In finetune_data.py, replace the process_and_merge function

def process_and_merge(subjects, source_2d, source_3d):
    """
    A robust function to process data from a source and convert it.
    It now safely handles subjects that might only exist in one of the files.
    """
    processed_2d = {}
    processed_3d = {}

    for subject in subjects:
        # --- NEW SAFETY CHECK ---
        # If the subject from the 2D data list isn't in the 3D data, skip it.
        if subject not in source_3d:
            print(f"WARNING: Subject {subject} found in 2D data but not in 3D data. Skipping.")
            continue
        # ------------------------

        processed_2d[subject], processed_3d[subject] = {}, {}
        for action in source_2d[subject].keys():
            if action not in source_3d[subject]:
                continue # Skip if action is not in both datasets

            poses_2d_action = source_2d[subject][action]
            poses_3d_action = source_3d[subject][action]
            
            # Use the minimum number of cameras to handle inconsistencies
            num_cameras = min(len(poses_2d_action), len(poses_3d_action))

            # Store the processed data for this action
            processed_2d[subject][action] = [poses_2d_action[i] for i in range(num_cameras)]
            processed_3d[subject][action] = [poses_3d_action[i] for i in range(num_cameras)]
            
    return processed_2d, processed_3d

if __name__ == '__main__':
    # --- 1. DEFINE FILE PATHS ---
    h36m_3d_path = 'data/data_3d_h36m.npz'
    h36m_2d_path = 'data/data_2d_h36m_cpn_ft_h36m_dbb.npz'
    mpi_root_path = 'data/mpi_inf_3dhp'

    output_3d_path = 'data/data_3d_merged.npz'
    output_2d_path = 'data/data_2d_merged.npz'
    # ---------------------------

    # --- 2. LOAD ORIGINAL HUMAN3.6M DATA ---
    print("Loading original Human3.6M data...")
    h36m_data_3d_orig = np.load(h36m_3d_path, allow_pickle=True)['positions_3d'].item()
    h36m_2d_npz = np.load(h36m_2d_path, allow_pickle=True)
    h36m_data_2d_orig = h36m_2d_npz['positions_2d'].item()
    h36m_metadata = h36m_2d_npz['metadata'].item()
    
    # Clean the H36M data to resolve inconsistencies
    print("Cleaning Human3.6M data...")
    h36m_data_2d, h36m_data_3d = process_and_merge(h36m_data_2d_orig.keys(), h36m_data_2d_orig, h36m_data_3d_orig)
    print("Human3.6M data loaded and cleaned.")

    # --- 3. PROCESS AND CONVERT MPI-INF-3DHP DATA ---
    print("\nProcessing MPI-INF-3DHP data...")
    subjects_to_process = ['S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    
    for subject_name in tqdm(subjects_to_process, desc="Processing Subjects"):
        subject_path = os.path.join(mpi_root_path, subject_name)
        new_subject_key = f"{subject_name}_MPI"
        h36m_data_3d[new_subject_key] = {}
        h36m_data_2d[new_subject_key] = {}

        sequence_folders = sorted([d for d in os.listdir(subject_path) if d.startswith('Seq')])
        for seq_name in sequence_folders:
            annot_path = os.path.join(subject_path, seq_name, 'annot.mat')
            if not os.path.exists(annot_path): continue

            annotations = sio.loadmat(annot_path)
            num_cameras = annotations['annot3'].shape[0]
            poses_3d_for_action, poses_2d_for_action = [], []

            for cam_idx in range(num_cameras):
                poses_3d_mpi = annotations['annot3'][cam_idx][0].reshape(-1, 28, 3)
                poses_2d_mpi = annotations['annot2'][cam_idx][0].reshape(-1, 28, 2)
                
                video_width, video_height = 2048, 2048
                poses_2d_mpi[..., :2] = normalize_screen_coordinates(poses_2d_mpi[..., :2], w=video_width, h=video_height)
                
                poses_3d_h36m = mpi_to_h36m_skeleton(poses_3d_mpi)
                poses_2d_h36m = mpi_to_h36m_skeleton(poses_2d_mpi)

                poses_3d_for_action.append(poses_3d_h36m)
                poses_2d_for_action.append(poses_2d_h36m)
            
            h36m_data_3d[new_subject_key][seq_name] = poses_3d_for_action
            h36m_data_2d[new_subject_key][seq_name] = poses_2d_for_action

    print("MPI-INF-3DHP processing complete.")

    # --- 4. SAVE THE MERGED .npz FILES ---
    print(f"\nSaving merged 3D data to {output_3d_path}...")
    np.savez_compressed(output_3d_path, positions_3d=h36m_data_3d)

    print(f"Saving merged 2D data to {output_2d_path}...")
    np.savez_compressed(output_2d_path, positions_2d=h36m_data_2d, metadata=h36m_metadata)

    print("\n✅ All done. Your merged dataset is clean and ready for fine-tuning.")