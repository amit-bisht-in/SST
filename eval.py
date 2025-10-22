# In eval.py

import numpy as np
from common.arguments import parse_args
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

# --- Project Imports ---
from common.model_sst import SST_Model
from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error
from common.h36m_dataset import Human36mDataset
from common.generators import UnchunkedGenerator
from common.utils import *
from common.camera import *

# --- Helper Functions ---
def add_motion_dynamics(sequence_2d):
    """Adds velocity channels to a 2D pose sequence."""
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    return np.concatenate((sequence_2d, velocity), axis=-1)

def fetch(subjects, keypoints, dataset_poses, action_filter=None, receptive_field=27):
    """
    Gathers all 2D and 3D pose sequences for a given list of subjects
    and filters out sequences that are too short.
    """
    out_poses_3d, out_poses_2d = [], []
    for subject in subjects:
        if subject not in keypoints or subject not in dataset_poses: continue
        for action in keypoints[subject].keys():
            if action not in dataset_poses[subject]: continue
            if action_filter is not None and not any(action.startswith(a) for a in action_filter): continue
            
            poses_2d_action, poses_3d_action = keypoints[subject][action], dataset_poses[subject][action]
            num_cameras = min(len(poses_2d_action), len(poses_3d_action))
            for i in range(num_cameras):
                seq_2d, seq_3d = poses_2d_action[i], poses_3d_action[i]
                if seq_2d.shape[0] < receptive_field or seq_3d.shape[0] < receptive_field: continue
                out_poses_2d.append(seq_2d)
                out_poses_3d.append(seq_3d)
    return out_poses_2d, out_poses_3d

def evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field):
    """
    Computes and prints all evaluation metrics for the test set.
    """
    epoch_loss_3d_pos, epoch_loss_3d_pos_procrustes, epoch_loss_3d_pos_scale, epoch_loss_3d_vel = 0, 0, 0, 0
    with torch.no_grad():
        model_pos.eval()
        all_predictions, all_ground_truth = [], []

        for _, batch_3d, batch_2d in tqdm(test_generator.next_epoch(), total=len(test_generator.poses_2d), desc="Evaluating"):
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).cuda()
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).cuda()
            num_frames = inputs_2d.shape[1]
            if num_frames < receptive_field: continue

            # Create overlapping windows for the model
            input_windows = [inputs_2d[:, i:i + receptive_field] for i in range(num_frames - receptive_field + 1)]
            input_windows = torch.cat(input_windows, dim=0)
            
            # Use mini-batches to avoid memory errors
            predictions_for_sequence = []
            for i in range(0, len(input_windows), 512):
                predicted_3d_pos_windows = model_pos(input_windows[i:i+512])
                predictions_for_sequence.append(predicted_3d_pos_windows.cpu().numpy())
            predicted_3d_pos_windows = np.concatenate(predictions_for_sequence)

            predicted_3d_pos = predicted_3d_pos_windows[:, receptive_field // 2]
            
            pad = (receptive_field - 1) // 2
            gt_3d = inputs_3d[:, pad:num_frames - pad].squeeze(0)
            gt_3d[:, 0] = 0
            
            all_predictions.append(predicted_3d_pos)
            all_ground_truth.append(gt_3d.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    ground_truth = np.concatenate(all_ground_truth)
    
    print("Saving predictions to predictions.npy...")
    np.save('predictions.npy', predictions)
    
    # Calculate all metrics
    N = predictions.shape[0]
    epoch_loss_3d_pos = mpjpe(torch.from_numpy(predictions), torch.from_numpy(ground_truth)) * N
    epoch_loss_3d_pos_procrustes = p_mpjpe(predictions, ground_truth) * N
    epoch_loss_3d_pos_scale = n_mpjpe(torch.from_numpy(predictions), torch.from_numpy(ground_truth)) * N
    if predictions.shape[0] > 1:
        epoch_loss_3d_vel = mean_velocity_error(predictions, ground_truth) * (N - 1)

    # Print the full report
    print('----------')
    print('Protocol #1 (MPJPE):', (epoch_loss_3d_pos / N) * 1000, 'mm')
    print('Protocol #2 (P-MPJPE):', (epoch_loss_3d_pos_procrustes / N) * 1000, 'mm')
    print('Protocol #3 (N-MPJPE):', (epoch_loss_3d_pos_scale / N) * 1000, 'mm')
    if epoch_loss_3d_vel > 0:
        print('Velocity Error (MPJVE):', (epoch_loss_3d_vel / (N - 1)) * 1000 if N > 1 else 0, 'mm/s')
    print('----------')

# --- Main Evaluation Logic ---
if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)
    
    # --- Data Loading and Preparation ---
    print(f"Loading '{args.dataset}' data for evaluation...")
    dataset = Human36mDataset('data/data_3d_' + args.dataset + '.npz')
    keypoints_data = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints = keypoints_data['positions_2d'].item()
    keypoints_metadata = keypoints_data['metadata'].item()
    kps_left, kps_right = list(keypoints_metadata['keypoints_symmetry'][0]), list(keypoints_metadata['keypoints_symmetry'][1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                if subject in dataset.cameras():
                    cam = dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                if kps.shape[-1] == 2:
                    keypoints[subject][action][cam_idx] = add_motion_dynamics(kps)

    # --- Model Creation ---
    print("Creating model...")
    num_joints = dataset.skeleton().num_joints()
    model_pos = SST_Model(args, num_joints=num_joints, in_chans=4)
    if torch.cuda.is_available(): model_pos = torch.nn.DataParallel(model_pos).cuda()
    print('INFO: Trainable parameter count:', sum(p.numel() for p in model_pos.parameters() if p.requires_grad))

    # --- Load Checkpoint ---
    if not args.evaluate:
        raise ValueError("--evaluate argument with a checkpoint path is required for this script")
        
    chk_filename = args.evaluate
    print(f"Loading checkpoint {chk_filename}")
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    # --- Run Evaluation ---
    receptive_field = args.number_of_frames
    pad = (receptive_field - 1) // 2
    subjects_test = args.subjects_test.split(',')
    action_filter = None if args.actions == '*' else args.actions.split(',')
    
    print(f"Fetching evaluation data for subjects: {subjects_test}")
    poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints, dataset._data, action_filter, receptive_field)
    
    test_generator = UnchunkedGenerator(None, poses_valid_3d, poses_valid_2d,
                                        pad=pad, kps_left=kps_left, kps_right=kps_right,
                                        joints_left=joints_left, joints_right=joints_right)
    
    evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field)

