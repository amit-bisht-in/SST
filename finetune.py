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
from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error, compute_total_loss
from common.h36m_dataset import Human36mDataset
from common.generators import ChunkedGenerator, UnchunkedGenerator
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
                if seq_2d.shape[0] < receptive_field or seq_3d.shape[0] < receptive_field:
                    continue
                out_poses_2d.append(seq_2d)
                out_poses_3d.append(seq_3d)
    
    if subset < 1.0 and len(out_poses_2d) > 0:
        num_sequences = len(out_poses_2d)
        num_to_sample = int(num_sequences * subset)
        indices = np.random.choice(num_sequences, num_to_sample, replace=False)
        out_poses_2d = [out_poses_2d[i] for i in indices]
        out_poses_3d = [out_poses_3d[i] for i in indices]
    return out_poses_2d, out_poses_3d

def evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field):
    # This function is used by the evaluation logic
    epoch_loss_3d_pos = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch_3d, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).cuda()
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).cuda()
            inputs_3d[:, :, 0] = 0
            predicted_3d_pos = model_pos(inputs_2d)
            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos += inputs_3d.shape[0] * error.item()
            N += inputs_3d.shape[0]
    final_error = (epoch_loss_3d_pos / N) * 10000
    print(f'Final Error (MPJPE): {final_error:.2f} mm')

# --- Main Script Logic ---
if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)

    try:
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST: raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    # --- Model Creation (Happens once, outside the if/else) ---
    print("Creating model...")
    # We load the base dataset just to get the skeleton definition
    base_dataset = Human36mDataset('data/data_3d_h36m.npz')
    num_joints = base_dataset.skeleton().num_joints()
    model_pos = SST_Model(args, num_joints=num_joints, in_chans=4)
    if torch.cuda.is_available(): model_pos = torch.nn.DataParallel(model_pos).cuda()
    print('INFO: Trainable parameter count:', sum(p.numel() for p in model_pos.parameters() if p.requires_grad))

    # --- Main Logic: Switch between Training and Evaluation ---
    if not args.evaluate:
        # --- TRAINING/FINE-TUNING LOGIC ---
        
        # 1. Load Skeleton and Metadata from Base H36M Dataset
        keypoints_metadata = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['metadata'].item()
        kps_left, kps_right = list(keypoints_metadata['keypoints_symmetry'][0]), list(keypoints_metadata['keypoints_symmetry'][1])
        joints_left, joints_right = list(base_dataset.skeleton().joints_left()), list(base_dataset.skeleton().joints_right())
        
        # 2. Load the MERGED Datasets for Fine-Tuning
        print('Loading MERGED 3D dataset for fine-tuning...')
        poses_3d_merged = np.load('data/data_3d_merged.npz', allow_pickle=True)['positions_3d'].item()
        print('Loading MERGED 2D detections for fine-tuning...')
        keypoints_merged = np.load('data/data_2d_merged.npz', allow_pickle=True)['positions_2d'].item()
        
        print("Preparing merged 2D data (adding velocity)...")
        for subject in keypoints_merged.keys():
            for action in keypoints_merged[subject]:
                for cam_idx, kps in enumerate(keypoints_merged[subject][action]):
                    keypoints_merged[subject][action][cam_idx] = add_motion_dynamics(kps)

       # In finetune.py, replace the entire "Setup for Fine-Tuning" section

        # --- 4. SETUP FOR FINE-TUNING ---
        receptive_field = args.number_of_frames
        subjects_train = args.subjects_train.split(',')
        subjects_test = args.subjects_test.split(',')
        action_filter = None if args.actions == '*' else args.actions.split(',')
        
        print("Fetching training and validation data from merged dataset...")
        poses_train_2d, poses_train_3d = fetch(subjects_train, keypoints_merged, poses_3d_merged, action_filter, receptive_field, subset=args.subset)
        poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints_merged, poses_3d_merged, action_filter, receptive_field)
        
        # Initialize optimizer and tracking variables
        optimizer = optim.AdamW(model_pos.parameters(), lr=args.learning_rate, weight_decay=0.1)
        lr_decay = args.lr_decay
        losses_3d_train, losses_3d_valid = [], []
        epoch = 0
        min_loss = 100000
        lr = args.learning_rate

        # --- THIS IS THE CORRECTED RESUME LOGIC ---
        if args.resume:
            if os.path.exists(args.resume):
                print(f"Loading checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
                
                # Load everything to properly resume the run
                model_pos.load_state_dict(checkpoint['model_pos'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                epoch = checkpoint['epoch']
                lr = checkpoint['lr']
                min_loss = checkpoint.get('min_loss', 100000) # Safely get min_loss
                
                print(f"Resuming fine-tuning from epoch {epoch + 1}")
            else:
                # If the specified resume file is not found, start a new run
                print(f"WARNING: Checkpoint file not found at '{args.resume}'. Starting new fine-tuning.")
        # ---------------------------------------------
                
        # Create the data generators
        pad = (receptive_field - 1) // 2
        train_generator = ChunkedGenerator(args.batch_size, None, poses_train_3d, poses_train_2d,
                                        chunk_length=receptive_field, pad=pad, shuffle=True,
                                        kps_left=kps_left, kps_right=kps_right,
                                        joints_left=joints_left, joints_right=joints_right,
                                        augment=args.data_augmentation, random_seed=epoch) # Use epoch for new random seed
        test_generator = ChunkedGenerator(args.batch_size, None, poses_valid_3d, poses_valid_2d,
                                        chunk_length=receptive_field, pad=pad, shuffle=False)

        print('** Starting fine-tuning...')
        while epoch < args.epochs:
            start_time = time()
            epoch_loss_3d_train, N = 0, 0
            model_pos.train()
           # Replace the for loop line with this entire block

# --- Logic for using a subset of batches per epoch ---
            if args.epoch_size is not None and args.epoch_size > 0 and args.epoch_size < train_generator.num_batches:
                num_batches_this_epoch = args.epoch_size
                iterator = itertools.islice(train_generator.next_epoch(), num_batches_this_epoch)
            else:
                num_batches_this_epoch = train_generator.num_batches
                iterator = train_generator.next_epoch()
            # -----------------------------------------------------------

            # The loop now uses our new iterator and total
            for _, batch_3d, batch_2d in tqdm(iterator, total=num_batches_this_epoch, desc=f"Finetune Epoch {epoch + 1}"):
                inputs_2d, inputs_3d = torch.from_numpy(batch_2d.astype('float32')).cuda(), torch.from_numpy(batch_3d.astype('float32')).cuda()
                inputs_3d[:, :, 0] = 0
                optimizer.zero_grad()
                predicted_3d_sequence = model_pos(inputs_2d)
                bones = [(p, c) for c, p in enumerate(base_dataset.skeleton().parents()) if p != -1]
                total_loss, loss_dict = compute_total_loss(predicted_3d_sequence, inputs_3d, bones)
                epoch_loss_3d_train += inputs_3d.shape[0] * loss_dict['pose_loss'].item()
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
           
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000000, losses_3d_valid[-1] * 1000000))
            
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
        print("This script is for fine-tuning. For evaluation, please use run_SST.py")