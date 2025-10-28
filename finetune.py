# In finetune.py

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

# --- Project Imports ---
from common.model_sst import SST_Model
from common.loss import mpjpe, compute_total_loss
from common.h36m_dataset import Human36mDataset
from common.fgenerator import ChunkedGenerator, UnchunkedGenerator
from common.utils import *
from common.camera import *

# --- Helper Functions ---
def add_motion_dynamics(sequence_2d):
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    return np.concatenate((sequence_2d, velocity), axis=-1)

def fetch(subjects, keypoints, dataset_poses, action_filter=None, receptive_field=27, subset=1.0):
    out_poses_3d, out_poses_2d = [], []
    for subject in subjects:
        if subject not in keypoints or subject not in dataset_poses:
            continue
        for action in keypoints[subject].keys():
            if action not in dataset_poses[subject]:
                continue
            if action_filter is not None and not any(action.startswith(a) for a in action_filter):
                continue
            
            poses_2d_action, poses_3d_action = keypoints[subject][action], dataset_poses[subject][action]
            num_cameras = min(len(poses_2d_action), len(poses_3d_action))
            
            for i in range(num_cameras):
                seq_2d, seq_3d = poses_2d_action[i], poses_3d_action[i]

                common_frames = min(seq_2d.shape[0], seq_3d.shape[0])
                seq_2d = seq_2d[:common_frames]
                seq_3d = seq_3d[:common_frames]

                if seq_2d.shape[0] < receptive_field:
                    continue
                out_poses_2d.append(seq_2d)
                out_poses_3d.append(seq_3d)
    
    if subset < 1.0 and len(out_poses_2d) > 0:
        num_sequences = len(out_poses_2d)
        num_to_sample = int(num_sequences * subset)
        random_state = np.random.RandomState()
        indices = random_state.choice(num_sequences, num_to_sample, replace=False)
        out_poses_2d = [out_poses_2d[i] for i in indices]
        out_poses_3d = [out_poses_3d[i] for i in indices]
    return out_poses_2d, out_poses_3d

def evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field):
    """
    Computes and prints all evaluation metrics. This version correctly uses the UnchunkedGenerator
    and a memory-efficient sliding window.
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

            input_windows = []
            for i in range(num_frames - receptive_field + 1):
                input_windows.append(inputs_2d[:, i:i + receptive_field])
            input_windows = torch.cat(input_windows, dim=0)
            
            predictions_for_sequence = []
            for i in range(0, len(input_windows), 512): # Process 512 windows at a time
                predicted_3d_pos_windows = model_pos(input_windows[i:i+512])
                predictions_for_sequence.append(predicted_3d_pos_windows.cpu().numpy())
            predicted_3d_pos_windows = np.concatenate(predictions_for_sequence)

            predicted_3d_pos = predicted_3d_pos_windows[:, receptive_field // 2]
            
            pad = (receptive_field - 1) // 2
            gt_3d = inputs_3d[:, pad:num_frames - pad].squeeze(0) # Get corresponding GT frames
            gt_3d[:, 0] = 0
            
            all_predictions.append(predicted_3d_pos)
            all_ground_truth.append(gt_3d.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    ground_truth = np.concatenate(all_ground_truth)
    
    print("Saving predictions to predictions.npy...")
    np.save('predictions.npy', predictions)
    
    N = predictions.shape[0]
    epoch_loss_3d_pos = mpjpe(torch.from_numpy(predictions), torch.from_numpy(ground_truth)) * N
    print('----------')
    print('Protocol #1 (MPJPE):', (epoch_loss_3d_pos / N) * 1000, 'mm')
    print('Protocol #2 (P-MPJPE):', (epoch_loss_3d_pos_procrustes / N) * 1000, 'mm')
    print('Protocol #3 (N-MPJPE):', (epoch_loss_3d_pos_scale / N) * 1000, 'mm')
    if epoch_loss_3d_vel > 0:
        print('Velocity Error (MPJVE):', (epoch_loss_3d_vel / (N - 1)) * 1000 if N > 1 else 0, 'mm/s')
    print('----------')

# --- Main Script Logic ---
if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)

    try:
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST: raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    # --- 1. Load Base Skeleton & Metadata (Used by both Train/Eval) ---
    print('Loading base H36M skeleton and metadata...')
    base_dataset = Human36mDataset('data/data_3d_h36m.npz')
    keypoints_metadata = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['metadata'].item()
    kps_left, kps_right = list(keypoints_metadata['keypoints_symmetry'][0]), list(keypoints_metadata['keypoints_symmetry'][1])
    joints_left, joints_right = list(base_dataset.skeleton().joints_left()), list(base_dataset.skeleton().joints_right())
    num_joints = base_dataset.skeleton().num_joints()

    # --- 2. Model Creation (Global Scope) ---
    print("Creating model...")
    model_pos = SST_Model(args, num_joints=num_joints, in_chans=4)
    if torch.cuda.is_available(): model_pos = torch.nn.DataParallel(model_pos).cuda()
    print('INFO: Trainable parameter count:', sum(p.numel() for p in model_pos.parameters() if p.requires_grad))

    # --- 3. Load Merged Data Dictionaries ---
    print(f"Loading MERGED 3D dataset: data/data_3d_merged.npz")
    poses_3d_merged = np.load('data/data_3d_merged.npz', allow_pickle=True)['positions_3d'].item()
    print(f"Loading MERGED 2D detections: data/data_2d_merged.npz")
    keypoints_merged = np.load('data/data_2d_merged.npz', allow_pickle=True)['positions_2d'].item()

    # --- 4. Main Logic: Switch between Training and Evaluation ---
    if not args.evaluate:
        # --- TRAINING/FINE-TUNING LOGIC ---
        
        print("Preparing merged 2D data (adding velocity)...")
        for subject in tqdm(keypoints_merged.keys(), desc="Adding Velocity"):
            for action in keypoints_merged[subject]:
                for cam_idx, kps in enumerate(keypoints_merged[subject][action]):
                    keypoints_merged[subject][action][cam_idx] = add_motion_dynamics(kps)
        
        receptive_field = args.number_of_frames
        subjects_train = args.subjects_train.split(',')
        subjects_test = args.subjects_test.split(',')
        action_filter = None if args.actions == '*' else args.actions.split(',')
        
        print("Fetching training and validation data from merged dataset...")
        poses_train_2d, poses_train_3d = fetch(subjects_train, keypoints_merged, poses_3d_merged, action_filter, receptive_field, subset=args.subset)
        poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints_merged, poses_3d_merged, action_filter, receptive_field)
        
        if not poses_valid_2d:
            print(f"ERROR: No valid validation data found for subjects {subjects_test}.")
            print(f"This can happen if video sequences are shorter than receptive field ({receptive_field} frames).")
            sys.exit()

        optimizer = optim.AdamW(model_pos.parameters(), lr=args.learning_rate, weight_decay=0.1)
        lr_decay = args.lr_decay
        losses_3d_train, losses_3d_valid, epoch, min_loss = [], [], 0, 100000
        lr = args.learning_rate

        # --- THIS IS THE UPDATED "SMART" RESUME LOGIC ---
        if args.resume:
            if os.path.exists(args.resume):
                print(f"Loading checkpoint: {args.resume}")
                # Load the full checkpoint (not weights_only)
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
                
                # Check if this is a "resume" or "start fine-tune" operation
                if args.resume.startswith(args.checkpoint):
                    # 1. RESUME a fine-tuning run
                    print("Optimizer state found in checkpoint. Resuming training...")
                    model_pos.load_state_dict(checkpoint['model_pos'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    epoch = checkpoint['epoch']
                    lr = checkpoint['lr']
                    min_loss = checkpoint.get('min_loss', 100000) # Get the score
                    print(f"Resuming fine-tuning from epoch {epoch + 1}")
                else:
                    # 2. START a new fine-tuning run (load weights only)
                    print("Checkpoint does not contain optimizer. Starting new fine-tuning session...")
                    # Create a new state dictionary to remove "module." prefix
                    new_state_dict = {}
                    for k, v in checkpoint['model_pos'].items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    
                    # Load the corrected weights into the model
                    model_pos.module.load_state_dict(new_state_dict)
                    print(f"Model weights from '{args.resume}' loaded successfully.")
                    
                    # RESET all training parameters for the new task
                    epoch = 0
                    lr = args.learning_rate # Use the new LR from the command
                    min_loss = 100000 # Reset loss tracker for new task
                    # Re-initialize the optimizer
                    optimizer = optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)
                    print(f"Resetting epoch to 1, loss to infinity, and using new LR: {lr}")
            else:
                print(f"WARNING: Checkpoint file not found at '{args.resume}'. Starting fine-tuning from scratch.")
        # ---------------------------------------------
                
        pad = (receptive_field - 1) // 2
        train_generator = ChunkedGenerator(args.batch_size, None, poses_train_3d, poses_train_2d,
                                           chunk_length=receptive_field, pad=pad, shuffle=True,
                                           kps_left=kps_left, kps_right=kps_right,
                                           joints_left=joints_left, joints_right=joints_right,
                                           augment=args.data_augmentation, random_seed=epoch)
        
        # This is the line that had the typo. It is now corrected.
        test_generator = ChunkedGenerator(args.batch_size, None, poses_valid_3d, poses_valid_2d,
                                          chunk_length=receptive_field, pad=pad, shuffle=False)
        
        print('** Starting fine-tuning...')
        while epoch < args.epochs:
            start_time = time()
            epoch_loss_3d_train, N = 0, 0
            model_pos.train()
            
            if args.epoch_size is not None and args.epoch_size > 0:
                num_batches_this_epoch = min(args.epoch_size, train_generator.num_batches)
                iterator = itertools.islice(train_generator.next_epoch(), num_batches_this_epoch)
            else:
                num_batches_this_epoch = train_generator.num_batches
                iterator = train_generator.next_epoch()

            for _, batch_3d, batch_2d in tqdm(iterator, total=num_batches_this_epoch, desc=f"Finetune Epoch {epoch + 1}"):
                inputs_2d, inputs_3d = torch.from_numpy(batch_2d.astype('float32')).cuda(), torch.from_numpy(batch_3d.astype('float32')).cuda()
                inputs_3d[:, :, 0] = 0
                optimizer.zero_grad()
                predicted_3d_sequence = model_pos(inputs_2d)
                bones = [(p, c) for c, p in enumerate(base_dataset.skeleton().parents()) if p != -1]
                total_loss = compute_total_loss(predicted_3d_sequence, inputs_3d)
                epoch_loss_3d_train += inputs_3d.shape[0] * total_loss.item()
                N += inputs_3d.shape[0]
                total_loss.backward()
                optimizer.step()
            losses_3d_train.append(epoch_loss_3d_train / N)
            
            with torch.no_grad():
                model_pos.eval()
                epoch_loss_3d_valid, N_valid = 0, 0
                for _, batch_3d_valid, batch_2d_valid in test_generator.next_epoch():
                    inputs_2d_valid, inputs_3d_valid = torch.from_numpy(batch_2d_valid.astype('float32')).cuda(), torch.from_numpy(batch_3d_valid.astype('float32')).cuda()
                    inputs_3d_valid[:, :, 0] = 0
                    predicted_3d_sequence_valid = model_pos(inputs_2d_valid)
                    valid_loss = mpjpe(predicted_3d_sequence_valid, inputs_3d_valid)
                    epoch_loss_3d_valid += inputs_3d_valid.shape[0] * valid_loss.item()
                    N_valid += inputs_3d_valid.shape[0]
                losses_3d_valid.append(epoch_loss_3d_valid / N_valid)

            elapsed = (time() - start_time) / 60
            # Corrected the print statement (was * 1000000)
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000, losses_3d_valid[-1] * 1000))
            
            lr *= lr_decay
            for param_group in optimizer.param_groups: param_group['lr'] *= lr_decay
            epoch += 1

            if losses_3d_valid[-1] < min_loss:
                min_loss = losses_3d_valid[-1]
                best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                print(f"New best model, saving checkpoint to {best_chk_path}")
                torch.save({'epoch': epoch, 'lr': lr, 'min_loss': min_loss,
                            'optimizer': optimizer.state_dict(), 'model_pos': model_pos.state_dict()}, best_chk_path)
    else:
        # --- EVALUATION LOGIC ---
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
        poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints_merged, poses_3d_merged, action_filter, receptive_field)
        
        # --- Optional: Run on a small random subset of test videos ---
        # print(f"Found {len(poses_valid_2d)} total test sequences.")
        # num_sequences_to_test = 2
        # if len(poses_valid_2d) > num_sequences_to_test:
        #     random_state = np.random.RandomState()
        #     indices = random_state.choice(len(poses_valid_2d), num_sequences_to_test, replace=False)
        #     poses_valid_2d = [poses_valid_2d[i] for i in indices]
        #     poses_valid_3d = [poses_valid_3d[i] for i in indices]
        # print(f"Running a quick, random evaluation on {len(poses_valid_2d)} sequences.")
        # ---------------------------------------------------------
        
        test_generator = UnchunkedGenerator(None, poses_valid_3d, poses_valid_2d,
                                            pad=pad, kps_left=kps_left, kps_right=kps_right,
                                            joints_left=joints_left, joints_right=joints_right)
        
        evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field)

