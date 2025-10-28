# In run_SST.py

import numpy as np
from common.arguments import parse_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import itertools 
from time import time
from tqdm import tqdm
import cv2
import collections
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving video
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse

# --- Project Imports ---
from common.model_sst import SST_Model
from common.loss import mpjpe, compute_total_loss
from common.h36m_dataset import Human36mDataset
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.utils import *
from common.camera import *

# --- NEW: Import for 2D Detector ---
from ultralytics import YOLO

# --- Helper Functions ---
def add_motion_dynamics(sequence_2d):
    """Adds velocity channels to a 2D pose sequence."""
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    return np.concatenate((sequence_2d, velocity), axis=-1)

def fetch(subjects, keypoints, dataset_poses, action_filter=None, receptive_field=27, subset=1.0):
    """
    Gathers all 2D and 3D pose sequences for a given list of subjects.
    It robustly filters out any sequences that are too short for the model's receptive field
    and can take a random subset of the training data if specified.
    """
    out_poses_3d, out_poses_2d = [], []
    for subject in subjects:
        # Safely check if subject exists in both data dictionaries
        if subject not in keypoints or subject not in dataset_poses:
            print(f"WARNING: Subject {subject} not found in one of the datasets. Skipping.")
            continue
        for action in keypoints[subject].keys():
            # Safely check if action exists in both data dictionaries
            if action not in dataset_poses[subject]:
                print(f"WARNING: Action {action} for {subject} not found in 3D data. Skipping.")
                continue
            if action_filter is not None and not any(action.startswith(a) for a in action_filter):
                continue

            # --- This logic handles the different data structures ---
            poses_2d_action_data = keypoints[subject][action]
            poses_3d_action_data = dataset_poses[subject][action]

            # The 2D data is a list of arrays
            poses_2d_action_list = list(poses_2d_action_data) if isinstance(poses_2d_action_data, dict) else poses_2d_action_data
            
            # The 3D data is a dictionary containing a 'positions_3d' list
            poses_3d_action_list = list(poses_3d_action_data['positions_3d'].values()) if isinstance(poses_3d_action_data['positions_3d'], dict) else poses_3d_action_data['positions_3d']
            # ----------------------------------------------------

            num_cameras = min(len(poses_2d_action_list), len(poses_3d_action_list))
            for i in range(num_cameras):
                seq_2d, seq_3d = poses_2d_action_list[i], poses_3d_action_list[i]
                if seq_2d.shape[0] < receptive_field or seq_3d.shape[0] < receptive_field:
                    continue
                out_poses_2d.append(seq_2d)
                out_poses_3d.append(seq_3d)
    
    # Apply subset sampling if requested
    if subset < 1.0 and len(out_poses_2d) > 0:
        num_sequences = len(out_poses_2d)
        num_to_sample = int(num_sequences * subset)
        # Use a new random state to ensure different subsets in each call
        random_state = np.random.RandomState()
        indices = random_state.choice(num_sequences, num_to_sample, replace=False)
        out_poses_2d = [out_poses_2d[i] for i in indices]
        out_poses_3d = [out_poses_3d[i] for i in indices]
        print(f"Using a random subset of {len(out_poses_2d)} training sequences ({subset*100:.0f}%).")

    return out_poses_2d, out_poses_3d


# --- NEW: Helper functions for video demo ---
def coco_to_h36m(keypoints):
    """Converts 17-joint COCO format keypoints to the 17-joint H36M format."""
    h36m_joints = np.zeros((17, 2), dtype=np.float32)
    h36m_joints[0] = (keypoints[11] + keypoints[12]) / 2
    h36m_joints[1], h36m_joints[2], h36m_joints[3] = keypoints[12], keypoints[14], keypoints[16]
    h36m_joints[4], h36m_joints[5], h36m_joints[6] = keypoints[11], keypoints[13], keypoints[15]
    h36m_joints[8] = (keypoints[5] + keypoints[6]) / 2
    h36m_joints[7] = (h36m_joints[0] + h36m_joints[8]) / 2
    h36m_joints[9] = (h36m_joints[8] * 0.75) + (keypoints[0] * 0.25)
    h36m_joints[10] = keypoints[0] + (keypoints[0] - h36m_joints[9])
    h36m_joints[11], h36m_joints[12], h36m_joints[13] = keypoints[5], keypoints[7], keypoints[9]
    h36m_joints[14], h36m_joints[15], h36m_joints[16] = keypoints[6], keypoints[8], keypoints[10]
    return h36m_joints

def draw_3d_skeleton(pose_3d, skeleton_parents, ax):
    """Draws a single 3D skeleton on a matplotlib Axes3D object."""
    pose_3d = pose_3d.copy()
    pose_3d[:, 2] *= -1 # Invert the Z-axis to flip upright
    pose_3d[:, 2] -= np.min(pose_3d[:, 2]) # Place the skeleton on the ground plane (z=0)

    ax.clear()
    ax.view_init(elev=15., azim=70)
    radius = 1000
    ax.set_xlim3d([-radius, radius]); ax.set_ylim3d([-radius, radius]); ax.set_zlim3d([0, radius * 1.5])
    ax.set_xlabel("X"); ax.set_ylabel("Y (Depth)"); ax.set_zlabel("Z (Height)")
    ax.set_title("3D Pose Reconstruction", pad=0)

    bones = [(p, c) for c, p in enumerate(skeleton_parents) if p != -1]
    for parent, child in bones:
        ax.plot([pose_3d[parent, 0], pose_3d[child, 0]],
                [pose_3d[parent, 1], pose_3d[child, 1]],
                [pose_3d[parent, 2], pose_3d[child, 2]], zdir='z', color='dodgerblue', linewidth=3)
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], zdir='z', c='red', s=20)

def fig_to_array(fig):
    """Converts a matplotlib figure to a numpy array (image)."""
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
# --- End of new helper functions ---

# In run_SST.py

def evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field):
    """
    Computes and prints all evaluation metrics. This version correctly uses the UnchunkedGenerator
    and a memory-efficient sliding window.
    """
    epoch_loss_3d_pos, epoch_loss_3d_pos_procrates, epoch_loss_3d_pos_scale, epoch_loss_3d_vel = 0, 0, 0, 0
    with torch.no_grad():
        model_pos.eval()
        all_predictions, all_ground_truth = [], []

        # This progress bar is now correct: it iterates over the number of videos
        for _, batch_3d, batch_2d in tqdm(test_generator.next_epoch(), total=len(test_generator.poses_2d), desc="Evaluating"):
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).cuda()
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).cuda()
            num_frames = inputs_2d.shape[1]
            if num_frames < receptive_field: continue

            # Create overlapping windows for the model from the long 2D sequence
            input_windows = []
            for i in range(num_frames - receptive_field + 1):
                input_windows.append(inputs_2d[:, i:i + receptive_field])
            input_windows = torch.cat(input_windows, dim=0)
            
            # Use mini-batches to avoid memory errors on long sequences
            predictions_for_sequence = []
            for i in range(0, len(input_windows), 512): # Process 512 windows at a time
                predicted_3d_pos_windows = model_pos(input_windows[i:i+512])
                predictions_for_sequence.append(predicted_3d_pos_windows.cpu().numpy())
            predicted_3d_pos_windows = np.concatenate(predictions_for_sequence)

            # Get center frame prediction for each window
            predicted_3d_pos = predicted_3d_pos_windows[:, receptive_field // 2]
            
            # --- THIS IS THE FIX ---
            # Create a matching ground_truth list by slicing the 3D data
            # in the exact same way the predictions were created.
            pad = (receptive_field - 1) // 2
            # Replace with this correct line
            gt_3d = inputs_3d.squeeze(0) # The GT is the entire unpadded sequence# Get corresponding GT frames
            gt_3d[:, 0] = 0
            # --------------------
            
            all_predictions.append(predicted_3d_pos)
            all_ground_truth.append(gt_3d.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    ground_truth = np.concatenate(all_ground_truth)
    
    # Save the final predictions for visualization
    print("Saving predictions to predictions.npy...")
    np.save('predictions.npy', predictions)
    
    # Calculate all metrics
    N = predictions.shape[0]
    epoch_loss_3d_pos = mpjpe(torch.from_numpy(predictions), torch.from_numpy(ground_truth)) * N

    # Print the full report
    print('----------')
    print('Protocol #1 (MPJPE):', (epoch_loss_3d_pos / N) * 1000, 'mm')
    if epoch_loss_3d_vel > 0:
        print('Velocity Error (MPJVE):', (epoch_loss_3d_vel / (N - 1)) * 1000 if N > 1 else 0, 'mm/s')
    print('----------')

# --- Main Script Logic ---
if __name__ == '__main__':
    # --- This is the fix: Add a specific parser for video arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to the video file for demo')
    
    # We parse *only* the new argument first.
    # The remaining arguments will be parsed by parse_args()
    vis_args, remaining_argv = parser.parse_known_args()

    # Pass the remaining arguments to the main parser
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_args()
    # --------------------------------------------------------

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)

    try:
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST: raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    # --- Data Loading (for all modes) ---
    print('Loading 3D dataset...')
    dataset = Human36mDataset('data/data_3d_' + args.dataset + '.npz')
    print('Loading 2D detections...')
    keypoints_data = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints = keypoints_data['positions_2d'].item()
    keypoints_metadata = keypoints_data['metadata'].item()
    kps_left, kps_right = list(keypoints_metadata['keypoints_symmetry'][0]), list(keypoints_metadata['keypoints_symmetry'][1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

    # --- Data Preparation and Synchronization ---
    print("Preparing data...")
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                if subject in dataset.cameras():
                    for cam in dataset.cameras()[subject]:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_3d[:, 1:] -= pos_3d[:, :1]
                        positions_3d.append(pos_3d)
                    anim['positions_3d'] = positions_3d

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if 'positions_3d' not in dataset[subject][action]: continue
            if subject not in keypoints or action not in keypoints[subject]: continue
            for cam_idx in range(len(keypoints[subject][action])):
                if cam_idx >= len(dataset[subject][action]['positions_3d']): continue
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                if subject in dataset.cameras() and cam_idx < len(dataset.cameras()[subject]):
                    cam = dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = add_motion_dynamics(kps)

    # --- Model Creation (Correct Global Scope) ---
    print("Creating model...")
    num_joints = dataset.skeleton().num_joints()
    model_pos = SST_Model(args, num_joints=num_joints, in_chans=4)
    if torch.cuda.is_available(): model_pos = torch.nn.DataParallel(model_pos).cuda()
    print('INFO: Trainable parameter count:', sum(p.numel() for p in model_pos.parameters() if p.requires_grad))

    # --- Main Logic: Switch between Training and Evaluation ---
    if not args.evaluate:
        # --- TRAINING LOGIC ---
        receptive_field = args.number_of_frames
        subjects_train, subjects_test = args.subjects_train.split(','), args.subjects_test.split(',')
        action_filter = None if args.actions == '*' else args.actions.split(',')
        
        print("Fetching validation data (this is done only once)...")
        poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints, dataset._data, action_filter, receptive_field)
        
        optimizer = optim.AdamW(model_pos.parameters(), lr=args.learning_rate, weight_decay=0.1)
        lr_decay = args.lr_decay
        losses_3d_train, losses_3d_valid, epoch, min_loss = [], [], 0, 100000
        lr = args.learning_rate

        if args.resume:
            if os.path.exists(args.resume):
                print(f"Loading checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage, weights_only=True)
                model_pos.load_state_dict(checkpoint['model_pos'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                epoch, lr, min_loss = checkpoint['epoch'], checkpoint['lr'], checkpoint.get('min_loss', 100000)
                print(f"Resuming training from epoch {epoch + 1}")
            else:
                print(f"WARNING: Checkpoint file not found at '{args.resume}'. Starting new training.")

        pad = (receptive_field - 1) // 2
        test_generator = ChunkedGenerator(args.batch_size, None, poses_valid_3d, poses_valid_2d,
                                          chunk_length=receptive_field, pad=pad, shuffle=False)

        print('** Note: reported losses are averaged over all frames.')
        while epoch < args.epochs:
            start_time = time()
            
            print(f"\nEpoch {epoch + 1}: Fetching new random training subset...")
            poses_train_2d, poses_train_3d = fetch(subjects_train, keypoints, dataset._data, action_filter, receptive_field, subset=args.subset)
            train_generator = ChunkedGenerator(args.batch_size, None, poses_train_3d, poses_train_2d,
                                               chunk_length=receptive_field, pad=pad, shuffle=True,
                                               kps_left=kps_left, kps_right=kps_right,
                                               joints_left=joints_left, joints_right=joints_right,
                                               augment=args.data_augmentation, random_seed=epoch)

            epoch_loss_3d_train, N = 0, 0
            model_pos.train()
            
            if args.epoch_size is not None and args.epoch_size > 0:
                num_batches_this_epoch = min(args.epoch_size, train_generator.num_batches)
                iterator = itertools.islice(train_generator.next_epoch(), num_batches_this_epoch)
            else:
                num_batches_this_epoch = train_generator.num_batches
                iterator = train_generator.next_epoch()

            for _, batch_3d, batch_2d in tqdm(iterator, total=num_batches_this_epoch, desc=f"Epoch {epoch + 1} Training"):
                inputs_2d, inputs_3d = torch.from_numpy(batch_2d.astype('float32')).cuda(), torch.from_numpy(batch_3d.astype('float32')).cuda()
                inputs_3d[:, :, 0] = 0
                optimizer.zero_grad()
                predicted_3d_sequence = model_pos(inputs_2d)
                bones = [(p, c) for c, p in enumerate(dataset.skeleton().parents()) if p != -1]
                total_loss = compute_total_loss(predicted_3d_sequence, inputs_3d)
                epoch_loss_3d_train += inputs_3d.shape[0] * total_loss.item()
                N += inputs_3d.shape[0]
                total_loss.backward()
                optimizer.step()
            losses_3d_train.append(epoch_loss_3d_train / N)
            
            with torch.no_grad():
                model_pos.eval()
                epoch_loss_3d_valid, N_valid = 0, 0
                for _, batch_3d_valid, batch_2d_valid in tqdm(test_generator.next_epoch(), total=test_generator.num_batches, desc=f"Epoch {epoch + 1} Validation"):
                    inputs_2d_valid, inputs_3d_valid = torch.from_numpy(batch_2d_valid.astype('float32')).cuda(), torch.from_numpy(batch_3d_valid.astype('float32')).cuda()
                    inputs_3d_valid[:, :, 0] = 0
                    predicted_3d_sequence_valid = model_pos(inputs_2d_valid)
                    valid_loss = mpjpe(predicted_3d_sequence_valid, inputs_3d_valid)
                    epoch_loss_3d_valid += inputs_3d_valid.shape[0] * valid_loss.item()
                    N_valid += inputs_3d_valid.shape[0]
                losses_3d_valid.append(epoch_loss_3d_valid / N_valid)

            elapsed = (time() - start_time) / 60
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000, losses_3d_valid[-1] * 1000))
            
            lr *= lr_decay
            for param_group in optimizer.param_groups: param_group['lr'] *= lr_decay
            epoch += 1

            if epoch % args.checkpoint_frequency == 0:
                chk_path = os.path.join(args.checkpoint, f'epoch_{epoch}.bin')
                print('Saving checkpoint to', chk_path)
                torch.save({'epoch': epoch, 'lr': lr, 'min_loss': min_loss, 'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos.state_dict()}, chk_path)
            
            if losses_3d_valid[-1] < min_loss:
                min_loss = losses_3d_valid[-1]
                best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                print(f"New best model, saving checkpoint to {best_chk_path}")
                torch.save({'epoch': epoch, 'lr': lr, 'min_loss': min_loss, 'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos.state_dict()}, best_chk_path)
    else:
        # --- NEW: Smart Evaluation Logic ---
        
        # Check if the user wants to run the video demo
        if vis_args.video:
            print(f"Running video demo on: {vis_args.video}")
            
            receptive_field = args.number_of_frames
            
            # --- Load 2D Detector (YOLO) ---
            print("Loading 2D YOLOv8-Pose model...")
            pose_model = YOLO('yolov8n-pose.pt') # Default model

            # --- Load 3D Model Checkpoint ---
            chk_filename = args.evaluate
            print(f"Loading checkpoint {chk_filename}")
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=True)
            new_state_dict = {}
            for k, v in checkpoint['model_pos'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            try:
                model_pos.module.load_state_dict(new_state_dict)
            except AttributeError:
                model_pos.load_state_dict(new_state_dict)
            model_pos.eval()

            # --- Initialize Video I/O ---
            cap = cv2.VideoCapture(vis_args.video)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)); fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_name = os.path.basename(vis_args.video)
            video_name_no_ext = os.path.splitext(video_name)[0]
            output_video_path = os.path.join(args.checkpoint, f"{video_name_no_ext}_3d_demo.mp4")
            
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))
            keypoints_buffer = collections.deque(maxlen=receptive_field)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            skeleton_parents = dataset.skeleton().parents()
            fig = plt.figure(figsize=(width/100, height/100))
            ax = fig.add_subplot(111, projection='3d')
            
            # --- Main Video Processing Loop ---
            for _ in tqdm(range(total_frames), desc="Processing Video"):
                ret, frame = cap.read()
                if not ret: break

                results = pose_model(frame, verbose=False)
                if len(results) > 0 and results[0].keypoints is not None and results[0].keypoints.shape[0] > 0:
                    keypoints_coco = results[0].keypoints.xy[0].cpu().numpy()
                    keypoints_h36m = coco_to_h36m(keypoints_coco)
                else:
                    keypoints_h36m = np.zeros((17, 2), dtype=np.float32)
                keypoints_buffer.append(keypoints_h36m)
                
                for joint in keypoints_h36m: cv2.circle(frame, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1)
                skeleton_img = np.zeros_like(frame) # Start with a black frame

                # Only start predicting and drawing when the buffer is full
                if len(keypoints_buffer) == receptive_field:
                    sequence_2d = np.array(keypoints_buffer, dtype=np.float32)
                    sequence_2d_normalized = normalize_screen_coordinates(sequence_2d, width, height)
                    sequence_4d = add_motion_dynamics(sequence_2d_normalized)
                    input_tensor = torch.from_numpy(sequence_4d).float().unsqueeze(0).cuda()

                    with torch.no_grad():
                        predicted_3d_sequence = model_pos(input_tensor)
                    predicted_3d_pose = predicted_3d_sequence[0, receptive_field // 2].cpu().numpy()
                    predicted_3d_pose *= 1000
                    
                    draw_3d_skeleton(predicted_3d_pose, skeleton_parents, ax)
                    skeleton_img = fig_to_array(fig)
                
                # Create side-by-side view and save
                combined_frame = np.hstack((frame, skeleton_img))
                out_video.write(combined_frame)

            cap.release(); out_video.release(); plt.close(fig)
            print(f"Processing finished. Output video saved to {output_video_path}")

        else:
            # --- ORIGINAL BENCHMARK EVALUATION LOGIC ---
            print(f"Running benchmark evaluation on dataset: {args.dataset}")
            
            # Load the checkpoint
            chk_filename = args.evaluate
            print(f"Loading checkpoint {chk_filename}")
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=True)
            
            new_state_dict = {}
            for k, v in checkpoint['model_pos'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            try:
                model_pos.module.load_state_dict(new_state_dict)
            except AttributeError:
                model_pos.load_state_dict(new_state_dict)

            receptive_field = args.number_of_frames
            pad = (receptive_field - 1) // 2
            subjects_test = args.subjects_test.split(',')
            action_filter = None if args.actions == '*' else args.actions.split(',')
            
            print(f"Fetching evaluation data for subjects: {subjects_test}")
            poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints, dataset._data, action_filter, receptive_field)
            
            # --- UPDATED BLOCK FOR A QUICK RANDOM TEST ---
            print(f"Found {len(poses_valid_2d)} total test sequences.")
            # Set a hardcoded number for the quick test
            num_sequences_to_test = 2 
            if len(poses_valid_2d) > num_sequences_to_test:
                # Use numpy to get a random sample
                random_state = np.random.RandomState()
                indices = random_state.choice(len(poses_valid_2d), num_sequences_to_test, replace=False)
                # Select the random subset
                poses_valid_2d = [poses_valid_2d[i] for i in indices]
                poses_valid_3d = [poses_valid_3d[i] for i in indices]
            print(f"Running a quick, random evaluation on {len(poses_valid_2d)} sequences.")
            # -------------------------------------------
            
            test_generator = UnchunkedGenerator(None, poses_valid_3d, poses_valid_2d,
                                                pad=pad, kps_left=kps_left, kps_right=kps_right,
                                                joints_left=joints_left, joints_right=joints_right)
            
            evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field)

