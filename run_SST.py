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
from time import time
from tqdm import tqdm

# --- Corrected Imports ---
from common.model_sst import SST_Model
from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error, compute_total_loss
from common.custom_dataset import CustomDataset
from common.h36m_dataset import Human36mDataset
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.utils import *
from common.camera import *

# --- Helper Functions ---
def add_motion_dynamics(sequence_2d):
    """Adds velocity channels to a 2D pose sequence."""
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    dynamic_sequence = np.concatenate((sequence_2d, velocity), axis=-1)
    return dynamic_sequence

# In run_SST.py, replace the entire fetch function

def fetch(subjects, keypoints, dataset, action_filter=None, receptive_field=27, subset=1.0):
    """
    Fetches 2D and 3D poses and robustly filters out short sequences.
    Also handles taking a random subset of the data.
    """
    out_poses_3d, out_poses_2d = [], []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None and not any(action.startswith(a) for a in action_filter):
                continue

            poses_2d_action = keypoints[subject][action]
            poses_3d_action = dataset[subject][action]['positions_3d']
            assert len(poses_3d_action) == len(poses_2d_action), 'Camera count mismatch'

            for i in range(len(poses_2d_action)):
                seq_2d, seq_3d = poses_2d_action[i], poses_3d_action[i]
                if seq_2d.shape[0] < receptive_field or seq_3d.shape[0] < receptive_field:
                    continue
                out_poses_2d.append(seq_2d)
                out_poses_3d.append(seq_3d)
                
    # --- NEW: Apply the subset sampling at the end ---
    if subset < 1.0:
        num_sequences = len(out_poses_2d)
        num_to_sample = int(num_sequences * subset)
        
        # Create random indices to sample
        indices = np.random.choice(num_sequences, num_to_sample, replace=False)
        
        # Select the subset
        out_poses_2d = [out_poses_2d[i] for i in indices]
        out_poses_3d = [out_poses_3d[i] for i in indices]
        print(f"Using a random subset of {len(out_poses_2d)} training sequences ({subset*100:.0f}%).")

    return out_poses_2d, out_poses_3d

# In run_SST.py
# In run_SST.py

# In run_SST.py

def evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    
    predictions = []
    ground_truth = []

    with torch.no_grad():
        model_pos.eval()
        # The generator now correctly yields only 3d and 2d poses for evaluation
        for _, batch_3d, batch_2d in tqdm(test_generator.next_epoch(), total=test_generator.num_frames(), desc="Evaluating"):
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).cuda()
            gt_3d = torch.from_numpy(batch_3d.astype('float32')).cuda()
            
            # Predict 3D poses (for the center frame)
            predicted_3d_pos = model_pos(inputs_2d)
            predicted_3d_pos = predicted_3d_pos[:, receptive_field // 2] # Get center frame prediction
            
            predictions.append(predicted_3d_pos.cpu().numpy())
            
            # --- THIS IS THE CORRECTED LINE ---
            # Match the prediction by also taking the center frame of the ground truth sequence
            ground_truth.append(gt_3d[:, receptive_field // 2].cpu().numpy())

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    
    # Now the shapes will match perfectly for metric calculation
    epoch_loss_3d_pos = mpjpe(torch.from_numpy(predictions), torch.from_numpy(ground_truth)) * predictions.shape[0]
    epoch_loss_3d_pos_procrustes = p_mpjpe(predictions, ground_truth) * predictions.shape[0]
    epoch_loss_3d_pos_scale = n_mpjpe(torch.from_numpy(predictions), torch.from_numpy(ground_truth)) * predictions.shape[0]
    epoch_loss_3d_vel = mean_velocity_error(predictions, ground_truth) * (predictions.shape[0] - 1)

    N = predictions.shape[0]

    # --- PRINT FULL REPORT ---
    print('----------')
    print('Protocol #1 (MPJPE):', (epoch_loss_3d_pos / N) * 1000, 'mm')
    print('Protocol #2 (P-MPJPE):', (epoch_loss_3d_pos_procrustes / N) * 1000, 'mm')
    print('Protocol #3 (N-MPJPE):', (epoch_loss_3d_pos_scale / N) * 1000, 'mm')
    print('Velocity Error (MPJVE):', (epoch_loss_3d_vel / (N-1)) * 1000 if N > 1 else 0, 'mm/s')
    print('----------')

# --- Main Script Logic ---
# --- Main Script Logic ---
args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)

try:
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

#-------------------------------------------
if not args.evaluate:
    # --- 1. DATA LOADING FOR TRAINING ---
    print('Loading 3D dataset for training...')
    dataset = Human36mDataset(path='data/data_3d_' + args.dataset + '.npz')

    print('Loading 2D detections for training...')
    keypoints_data = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints_data['metadata'].item()
    kps_left, kps_right = list(keypoints_metadata['keypoints_symmetry'][0]), list(keypoints_metadata['keypoints_symmetry'][1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints_data['positions_2d'].item()

    # --- 2. DATA PREPARATION ---
    print("Preparing data...")
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if 'positions_3d' not in dataset[subject][action]: continue
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = add_motion_dynamics(kps)

    # --- 3. MODEL CREATION (Happens only ONCE) ---
    print("Creating model...")
    num_joints = dataset.skeleton().num_joints()
    model_pos = SST_Model(args, num_joints=num_joints, in_chans=4)
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos).cuda()
    print('INFO: Trainable parameter count:', sum(p.numel() for p in model_pos.parameters() if p.requires_grad))

    # --- 4. SETUP FOR TRAINING ---
    receptive_field = args.number_of_frames
    subjects_train = args.subjects_train.split(',')

    print(f"DEBUG: Attempting to train on subjects: {subjects_train}") # <-- ADD THIS LINE
    subjects_test = args.subjects_test.split(',')
    action_filter = None if args.actions == '*' else args.actions.split(',')

    print("Fetching training and validation data...")
    poses_train_2d, poses_train_3d = fetch(subjects_train, keypoints, dataset, action_filter, receptive_field)
    poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints, dataset, action_filter, receptive_field)

    optimizer = optim.AdamW(model_pos.parameters(), lr=args.learning_rate, weight_decay=0.1)
    lr_decay = args.lr_decay
    losses_3d_train, losses_3d_valid = [], []

    if args.resume:
        chk_filename = args.resume
        print(f"Loading checkpoint {chk_filename}")
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        min_loss = checkpoint.get('min_loss', 100000)
        print(f"Resuming training from epoch {epoch + 1}")
    else:
        epoch = 0
        lr = args.learning_rate
        min_loss = 100000

    pad = (receptive_field - 1) // 2
    train_generator = ChunkedGenerator(args.batch_size, None, poses_train_3d, poses_train_2d,
                                     chunk_length=receptive_field, pad=pad, shuffle=True,
                                     kps_left=kps_left, kps_right=kps_right,
                                     joints_left=joints_left, joints_right=joints_right,
                                     augment=args.data_augmentation)
    test_generator = ChunkedGenerator(args.batch_size, None, poses_valid_3d, poses_valid_2d,
                                      chunk_length=receptive_field, pad=pad, shuffle=False)

    # --- 5. MAIN TRAINING LOOP ---
    print('** Note: reported losses are averaged over all frames.')
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train, N = 0, 0
        model_pos.train()

        # Loop over training batches
        for _, batch_3d, batch_2d in tqdm(train_generator.next_epoch(), total=train_generator.num_batches, desc=f"Epoch {epoch + 1}"):
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).cuda()
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).cuda()
            inputs_3d[:, :, 0] = 0
            
            optimizer.zero_grad()
            
            predicted_3d_sequence = model_pos(inputs_2d)
            
            bones = [(p, c) for c, p in enumerate(dataset.skeleton().parents()) if p != -1]
            total_loss, loss_dict = compute_total_loss(predicted_3d_sequence, inputs_3d, bones)
            
            epoch_loss_3d_train += inputs_3d.shape[0] * loss_dict['pose_loss'].item()
            N += inputs_3d.shape[0]
            
            total_loss.backward()
            optimizer.step()
            
        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.eval()
            epoch_loss_3d_valid, N_valid = 0, 0
            for _, batch_3d_valid, batch_2d_valid in test_generator.next_epoch():
                inputs_2d_valid = torch.from_numpy(batch_2d_valid.astype('float32')).cuda()
                inputs_3d_valid = torch.from_numpy(batch_3d_valid.astype('float32')).cuda()
                inputs_3d_valid[:, :, 0] = 0
                
                predicted_3d_sequence_valid = model_pos(inputs_2d_valid)
                valid_loss = mpjpe(predicted_3d_sequence_valid, inputs_3d_valid)
                
                epoch_loss_3d_valid += inputs_3d_valid.shape[0] * valid_loss.item()
                N_valid += inputs_3d_valid.shape[0]
                
            losses_3d_valid.append(epoch_loss_3d_valid / N_valid)

        elapsed = (time() - start_time) / 60
        print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' %
              (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000, losses_3d_valid[-1] * 1000))

        # Decay learning rate
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Save checkpoints
        if losses_3d_valid[-1] < min_loss:
            min_loss = losses_3d_valid[-1]
            best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
            print(f"New best model, saving checkpoint to {best_chk_path}")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'min_loss': min_loss,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
            }, best_chk_path)

        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, f'epoch_{epoch}.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'min_loss': min_loss,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
            }, chk_path)

else:
    # --- EFFICIENT EVALUATION LOGIC ---
    print('Evaluating...')
    
    subjects_test = args.subjects_test.split(',')
    print(f"Loading data for test subjects only: {subjects_test}")

    # Efficiently load and filter 3D data
    print('Loading 3D dataset...')
    data_3d_all = np.load('data/data_3d_' + args.dataset + '.npz', allow_pickle=True)['positions_3d'].item()
    data_3d_test = {subject: data_3d_all[subject] for subject in subjects_test}
    dataset = Human36mDataset(data=data_3d_test)

    # Efficiently load and filter 2D detections
    print('Loading 2D detections...')
    keypoints_all_data = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints_all_data['metadata'].item()
    kps_left, kps_right = list(keypoints_metadata['keypoints_symmetry'][0]), list(keypoints_metadata['keypoints_symmetry'][1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    
    keypoints_all_2d = keypoints_all_data['positions_2d'].item()
    keypoints = {subject: keypoints_all_2d[subject] for subject in subjects_test}

    # --- START: ADD THIS MISSING DATA PREPARATION BLOCK ---
    print("Preparing data...")
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if 'positions_3d' not in dataset[subject][action]: continue
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = add_motion_dynamics(kps)
    # --- END: OF MISSING DATA PREPARATION BLOCK ---

    # --- Model Creation ---
    print("Creating model...")
    num_joints = dataset.skeleton().num_joints()
    model_pos = SST_Model(args, num_joints=num_joints, in_chans=4)
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos).cuda()
    print('INFO: Trainable parameter count:', sum(p.numel() for p in model_pos.parameters() if p.requires_grad))

    # --- Load Checkpoint & Run Evaluation ---
    chk_filename = args.evaluate
    print(f"Loading checkpoint {chk_filename}")
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    receptive_field = args.number_of_frames
    pad = (receptive_field - 1) // 2
    action_filter = None if args.actions == '*' else args.actions.split(',')
    
    print("Fetching evaluation data...")
    poses_valid_2d, poses_valid_3d = fetch(subjects_test, keypoints, dataset, action_filter, receptive_field)
    
    test_generator = ChunkedGenerator(args.batch_size, None, poses_valid_3d, poses_valid_2d,
                                      chunk_length=receptive_field, pad=pad, shuffle=False)
    
    evaluate(test_generator, model_pos, joints_left, joints_right, kps_left, kps_right, receptive_field)