# In prepare_finetune_dataset.py

import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path to allow importing from `common`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from common.camera import world_to_camera, normalize_screen_coordinates
    from common.h36m_dataset import Human36mDataset
except ImportError:
    print("Error: Could not import from common. Make sure this script is in the root SST folder.")
    sys.exit()

def mpi_to_h36m_skeleton(poses, is_3d=True):
    """
    Converts a 28-joint MPI-INF-3DHP pose sequence to a 17-joint H36M pose sequence.
    """
    # Determine the number of coordinates (2 for 2D, 3 for 3D)
    num_coords = 3 if is_3d else 2
    h36m_poses = np.zeros((poses.shape[0], 17, num_coords), dtype=np.float32)

    # MPI-INF-3DHP joint indices
    MPI_Spine, MPI_Thorax, MPI_Neck, MPI_Head, MPI_L_Shoulder, MPI_L_Elbow, MPI_L_Wrist, \
    MPI_R_Shoulder, MPI_R_Elbow, MPI_R_Wrist, MPI_L_Hip, MPI_L_Knee, MPI_L_Ankle, \
    MPI_R_Hip, MPI_R_Knee, MPI_R_Ankle = 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18

    # H36M joint indices
    H36M_Pelvis, H36M_R_Hip, H36M_R_Knee, H36M_R_Ankle, H36M_L_Hip, H36M_L_Knee, H36M_L_Ankle, \
    H36M_Spine, H36M_Thorax, H36M_Neck, H36M_Head, H36M_L_Shoulder, H36M_L_Elbow, H36M_L_Wrist, \
    H36M_R_Shoulder, H36M_R_Elbow, H36M_R_Wrist = range(17)

    # --- Perform the mapping ---
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
    h36m_poses[:, H36M_Spine] = poses[:, MPI_Spine]
    h36m_poses[:, H36M_Pelvis] = (poses[:, MPI_L_Hip] + poses[:, MPI_R_Hip]) / 2

    return h36m_poses

def process_h36m_data(subjects, source_2d, source_3d_dataset_obj):
    """
    Processes the original H36M data, converting 3D poses to camera space
    and ensuring data is consistent.
    """
    processed_2d = {}
    processed_3d = {}

    for subject in subjects:
        if subject not in source_3d_dataset_obj._data:
            print(f"WARNING: Subject {subject} found in 2D data but not in 3D data. Skipping.")
            continue
            
        processed_2d[subject], processed_3d[subject] = {}, {}
        
        for action in source_2d[subject].keys():
            if action not in source_3d_dataset_obj._data[subject]:
                print(f"WARNING: Action {action} for {subject} not found in 3D data. Skipping.")
                continue

            poses_2d_action_list = source_2d[subject][action]
            
            # Manually perform the camera space conversion for H36M data
            anim = source_3d_dataset_obj._data[subject][action]
            poses_3d_action_list = []
            if 'positions' in anim:
                for cam in source_3d_dataset_obj._cameras[subject]:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # This is the original root-relative conversion for H36M
                    pos_3d[:, 1:] -= pos_3d[:, :1] 
                    poses_3d_action_list.append(pos_3d)
            else:
                if 'positions_3d' in anim:
                     poses_3d_action_list = anim['positions_3d']
                else:
                    print(f"WARNING: 'positions' key missing for {subject}/{action}. Skipping.")
                    continue

            num_cameras = min(len(poses_2d_action_list), len(poses_3d_action_list))

            processed_2d[subject][action] = [poses_2d_action_list[i] for i in range(num_cameras)]
            processed_3d[subject][action] = [poses_3d_action_list[i] for i in range(num_cameras)]
                
    return processed_2d, processed_3d

if __name__ == '__main__':
    # --- 1. DEFINE FILE PATHS ---
    h36m_3d_path = 'data/data_3d_h36m.npz'
    h36m_2d_path = 'data/data_2d_h36m_cpn_ft_h36m_dbb.npz'
    mpi_root_path = 'data/mpi_inf_3dhp'

    output_3d_path = 'data/data_3d_merged.npz'
    output_2d_path = 'data/data_2d_merged.npz'

    # --- 2. LOAD AND PROCESS HUMAN3.6M DATA ---
    print("Loading original Human3.6M data...")
    h36m_dataset_obj = Human36mDataset(h36m_3d_path)
    
    h36m_2d_npz = np.load(h36m_2d_path, allow_pickle=True)
    h36m_data_2d_orig = h36m_2d_npz['positions_2d'].item()
    h36m_metadata = h36m_2d_npz['metadata'].item()
    
    print("Cleaning and processing Human3.6M data...")
    h36m_subjects = h36m_data_2d_orig.keys()
    h36m_data_2d, h36m_data_3d = process_h36m_data(h36m_subjects, h36m_data_2d_orig, h36m_dataset_obj)
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

            try:
                annotations = sio.loadmat(annot_path)
            except Exception as e:
                print(f"WARNING: Could not load {annot_path}. Error: {e}. Skipping.")
                continue

            num_cameras = annotations['annot3'].shape[0]
            poses_3d_for_action, poses_2d_for_action = [], []

            for cam_idx in range(num_cameras):
                poses_3d_mpi = annotations['annot3'][cam_idx][0].reshape(-1, 28, 3)
                poses_2d_mpi = annotations['annot2'][cam_idx][0].reshape(-1, 28, 2)
                
                video_width, video_height = 2048, 2048
                poses_2d_mpi[..., :2] = normalize_screen_coordinates(poses_2d_mpi[..., :2], w=video_width, h=video_height)
                
                poses_3d_h36m = mpi_to_h36m_skeleton(poses_3d_mpi, is_3d=True)
                poses_2d_h36m = mpi_to_h36m_skeleton(poses_2d_mpi, is_3d=False)
                
                # --- THIS IS THE FIX ---
                # Make the MPI-INF-3DHP data root-relative, just like the H36M data
                # H36M_Pelvis is joint index 0
                poses_3d_h36m[:, 1:, :] -= poses_3d_h36m[:, :1, :]
                # --------------------

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

