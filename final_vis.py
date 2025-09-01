# In a new file: final_demo.py

import torch
import numpy as np
import cv2
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import sys
import argparse

# --- Project Imports ---
from common.camera import normalize_screen_coordinates
from common.model_sst import SST_Model
from common.h36m_dataset import Human36mDataset

# --- Ultralytics Import for 2D Pose Detection ---
from ultralytics import YOLO

# --- Helper Functions ---
def add_motion_dynamics(sequence_2d):
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    return np.concatenate((sequence_2d, velocity), axis=-1)

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

# --- Main Application Logic ---

def main(cli_args):
    receptive_field = 27

    # --- Load Models ---
    print("Loading 3D SST_Model...")
    model_args = type('Args', (object,), {'number_of_frames': receptive_field, 'number_of_kept_frames': receptive_field, 
                                          'number_of_kept_coeffs': receptive_field, 'depth': 4, 'dropout': 0.1, 
                                          'embed_dim_ratio': 32, 'n_heads': 8})
    sst_model = SST_Model(model_args, num_joints=17, in_chans=4)
    if torch.cuda.is_available(): sst_model = torch.nn.DataParallel(sst_model).cuda()
    checkpoint = torch.load(cli_args.checkpoint_3d, map_location=lambda storage, loc: storage)
    sst_model.load_state_dict(checkpoint['model_pos'])
    sst_model.eval()

    print("Loading 2D YOLOv8-Pose model...")
    pose_model = YOLO('yolov8n-pose.pt') 

    # --- Pass 1: Detect all 2D poses ---
    cap = cv2.VideoCapture(cli_args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_2d_poses = []
    print("Performing 2D pose detection on video...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        results = pose_model(frame, verbose=False)
        if len(results) > 0 and results[0].keypoints.shape[0] > 0:
            keypoints_coco = results[0].keypoints.xy[0].cpu().numpy()
            keypoints_h36m = coco_to_h36m(keypoints_coco)
        else:
            keypoints_h36m = np.zeros((17, 2), dtype=np.float32)
        all_2d_poses.append(keypoints_h36m)
    cap.release()
    
    # --- Pass 2: Setup Interactive Animation ---
    print("Setting up 3D animation...")
    dataset = Human36mDataset('data/data_3d_h36m.npz')
    skeleton_parents = dataset.skeleton().parents()
    bones = [(p, c) for c, p in enumerate(skeleton_parents) if p != -1]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=70)
    radius = 1000

    # --- CORRECTED PLOT SETUP for Z-up ---
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])   # Y is now depth
    ax.set_zlim3d([0, radius * 1.5])     # Z is now vertical (height)
    ax.set_xlabel("X"); ax.set_ylabel("Y (Depth)"); ax.set_zlabel("Z (Height)")
    # ------------------------------------

    lines = [ax.plot([], [], [], 'dodgerblue', linewidth=3)[0] for _ in bones]
    points = ax.scatter([], [], [], c='red', s=20)
    keypoints_buffer = collections.deque(maxlen=receptive_field)

    def update(frame_index):
        keypoints_buffer.append(all_2d_poses[frame_index])
        pose_3d = np.zeros((17, 3))

        if len(keypoints_buffer) == receptive_field:
            sequence_2d = np.array(keypoints_buffer, dtype=np.float32)
            sequence_2d_normalized = normalize_screen_coordinates(sequence_2d, width, height)
            sequence_4d = add_motion_dynamics(sequence_2d_normalized)
            input_tensor = torch.from_numpy(sequence_4d).float().unsqueeze(0).cuda()

            with torch.no_grad():
                predicted_3d_sequence = sst_model(input_tensor)
            
            pose_3d = predicted_3d_sequence[0, receptive_field // 2].cpu().numpy()
            pose_3d *= 1000
        
        # --- THIS IS THE TRANSFORMATION FOR THE "LYING DOWN" SKELETON ---
        pose_3d_oriented = np.zeros_like(pose_3d)
        # X stays the same
        pose_3d_oriented[:, 0] = pose_3d[:, 0]
        # New Y (depth) is the original Z (height)
        pose_3d_oriented[:, 1] = pose_3d[:, 2]
        # New Z (height) is the original Y (depth)
        pose_3d_oriented[:, 2] = pose_3d[:, 1]
        
        # Center the skeleton vertically
        pose_3d_oriented[:, 2] -= np.mean(pose_3d_oriented[:, 2])
        # ----------------------------------------------------------------

        ax.set_title(f"Interactive 3D Pose (Frame {frame_index})")
        for i, (parent, child) in enumerate(bones):
            lines[i].set_data([pose_3d_oriented[parent, 0], pose_3d_oriented[child, 0]], [pose_3d_oriented[parent, 1], pose_3d_oriented[child, 1]])
            lines[i].set_3d_properties([pose_3d_oriented[parent, 2], pose_3d_oriented[child, 2]])
        points._offsets3d = (pose_3d_oriented[:, 0], pose_3d_oriented[:, 1], pose_3d_oriented[:, 2])
        
        return lines + [points]

    anim = animation.FuncAnimation(fig, update, frames=len(all_2d_poses), interval=40, blit=False)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SST Model Interactive Video Demo')
    parser.add_argument('--checkpoint-3d', type=str, default='checkpoint/SST_Model_final/best_epoch.bin', help='Path to the trained 3D model checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    cli_args = parser.parse_args()
    main(cli_args)