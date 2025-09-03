# In prepare_finetune_dataset.py

import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm

def mpi_to_h36m_skeleton(poses):
    """
    Converts a 28-joint MPI-INF-3DHP pose sequence to a 17-joint H36M pose sequence.
    Args:
        poses (np.ndarray): A NumPy array of shape (num_frames, 28, 2) or (num_frames, 28, 3).
    Returns:
        np.ndarray: A NumPy array of shape (num_frames, 17, 3).
    """
    # The new array for the 17-joint H36M skeleton
    h36m_poses = np.zeros((poses.shape[0], 17, poses.shape[2]), dtype=np.float32)

    # MPI-INF-3DHP joint indices
    MPI_Spine, MPI_Thorax, MPI_Neck, MPI_Head, MPI_L_Shoulder, MPI_L_Elbow, MPI_L_Wrist, \
    MPI_R_Shoulder, MPI_R_Elbow, MPI_R_Wrist, MPI_L_Hip, MPI_L_Knee, MPI_L_Ankle, \
    MPI_R_Hip, MPI_R_Knee, MPI_R_Ankle = 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18

    # H36M joint indices
    H36M_Pelvis, H36M_R_Hip, H36M_R_Knee, H36M_R_Ankle, H36M_L_Hip, H36M_L_Knee, H36M_L_Ankle, \
    H36M_Spine, H36M_Thorax, H36M_Neck, H36M_Head, H36M_L_Shoulder, H36M_L_Elbow, H36M_L_Wrist, \
    H36M_R_Shoulder, H36M_R_Elbow, H36M_R_Wrist = range(17)

    # --- Perform the mapping ---
    # Direct mappings
    h36m_poses[:, H36M_R_Hip] = poses[:, MPI_R_Hip]
    h36m_poses[:, H36M_R_Knee] = poses[:, MPI_R_Knee]
    h36m_poses[:, H36M_R_Ankle] = poses[:, MPI_R_Ankle]
    h36m_poses[:, H36M_L_Hip] = poses[:, MPI_L_Hip]
    h36m_poses[:, H36M_L_Knee] = poses[:, MPI_L_Knee]
    h36m_poses[:, H36M_L_Ankle] = poses[:, MPI_L_Ankle]
    
    h36m_poses[:, H36M_R_Shoulder] = poses[:, MPI_R_Shoulder]
    h36m_poses[:, H36M_R_Elbow] = poses[:, MPI_R_Elbow]
    h36m_poses[:, H36M_R_Wrist] = poses[:, MPI_R_Wrist]
    h36m_poses[:, H36M_L_Shoulder] = poses[:, MPI_L_Shoulder]
    h36m_poses[:, H36M_L_Elbow] = poses[:, MPI_L_Elbow]
    h36m_poses[:, H36M_L_Wrist] = poses[:, MPI_L_Wrist]
    
    h36m_poses[:, H36M_Head] = poses[:, MPI_Head]
    h36m_poses[:, H36M_Neck] = poses[:, MPI_Neck]
    h36m_poses[:, H36M_Thorax] = poses[:, MPI_Thorax]
    
    # Calculated joints
    h36m_poses[:, H36M_Spine] = poses[:, MPI_Spine]
    h36m_poses[:, H36M_Pelvis] = (poses[:, MPI_L_Hip] + poses[:, MPI_R_Hip]) / 2

    return h36m_poses

if __name__ == '__main__':
    # --- 1. DEFINE FILE PATHS ---
    h36m_3d_path = r'data\data_3d_h36m.npz'
    h36m_2d_path = r'data/data_2d_h36m_cpn_ft_h36m_dbb.npz'
    mpi_root_path = r'data/mpi_inf_3dhp'

    # Output paths for the new merged files
    output_3d_path = 'data/data_3d_merged.npz'
    output_2d_path = 'data/data_2d_merged.npz'
    # ---------------------------

    # --- 2. LOAD EXISTING HUMAN3.6M DATA ---
    print("Loading original Human3.6M data...")
    h36m_data_3d = np.load(h36m_3d_path, allow_pickle=True)['positions_3d'].item()
    h36m_data_2d = np.load(h36m_2d_path, allow_pickle=True)['positions_2d'].item()
    print("Human3.6M data loaded.")
    # ------------------------------------

    # --- 3. PROCESS AND CONVERT MPI-INF-3DHP DATA ---
    print("\nProcessing MPI-INF-3DHP data...")
    subjects_to_process = ['S3', 'S4', 'S5', 'S6', 'S7', 'S8'] # Excluding the corrupted S1 and S2
    
    for subject_name in tqdm(subjects_to_process, desc="Processing Subjects"):
        subject_path = os.path.join(mpi_root_path, subject_name)
        
        # We'll create new subject keys to avoid clashes, e.g., 'S3_MPI'
        new_subject_key_3d = f"{subject_name}_MPI"
        h36m_data_3d[new_subject_key_3d] = {}
        h36m_data_2d[new_subject_key_3d] = {}

        sequence_folders = sorted([d for d in os.listdir(subject_path) if d.startswith('Seq')])
        for seq_name in sequence_folders:
            annot_path = os.path.join(subject_path, seq_name, 'annot.mat')
            if not os.path.exists(annot_path): continue

            annotations = sio.loadmat(annot_path)
            
            # The data is structured per camera view
            num_cameras = annotations['annot3'].shape[0]
            
            # Create lists to hold the data for each camera in this sequence
            poses_3d_for_action = []
            poses_2d_for_action = []

            for cam_idx in range(num_cameras):
                # Reshape flattened arrays
                poses_3d_flat = annotations['annot3'][cam_idx][0]
                poses_3d_mpi = poses_3d_flat.reshape(-1, 28, 3)
                
                poses_2d_flat = annotations['annot2'][cam_idx][0]
                poses_2d_mpi = poses_2d_flat.reshape(-1, 28, 2)

                # Convert skeletons to H36M format
                poses_3d_h36m = mpi_to_h36m_skeleton(poses_3d_mpi)
                poses_2d_h36m = mpi_to_h36m_skeleton(poses_2d_mpi)

                poses_3d_for_action.append(poses_3d_h36m)
                poses_2d_for_action.append(poses_2d_h36m)
            
            # Add this sequence (with all its camera views) to our main dictionary
            h36m_data_3d[new_subject_key_3d][seq_name] = poses_3d_for_action
            h36m_data_2d[new_subject_key_3d][seq_name] = poses_2d_for_action

    print("MPI-INF-3DHP processing complete.")
    # -----------------------------------------------

    # --- 4. SAVE THE MERGED .npz FILES ---
    print(f"\nSaving merged 3D data to {output_3d_path}...")
    np.savez_compressed(output_3d_path, positions_3d=h36m_data_3d)

    print(f"Saving merged 2D data to {output_2d_path}...")
    np.savez_compressed(output_2d_path, positions_2d=h36m_data_2d)

    print("\n✅ All done. Your merged dataset is ready for fine-tuning.")
    # ------------------------------------