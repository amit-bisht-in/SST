# In vis.py

import torch
import numpy as np
import cv2
import collections
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving video frames
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse

# --- Project Imports ---
# This allows the script to find your custom modules in the 'common' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.arguments import parse_args
from common.camera import normalize_screen_coordinates
from common.model_sst import SST_Model
from common.h36m_dataset import Human36mDataset

# --- Ultralytics Import for 2D Pose Detection ---
from ultralytics import YOLO

# --- Helper Functions ---

def add_motion_dynamics(sequence_2d):
    """Calculates velocity and adds it as two new channels to the 2D pose sequence."""
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    return np.concatenate((sequence_2d, velocity), axis=-1)

def coco_to_h36m(keypoints):
    """
    Converts 17-joint COCO format keypoints to the 17-joint H36M format
    that our 3D model was trained on.
    """
    h36m_joints = np.zeros((17, 2), dtype=np.float32)

    # Mapping based on joint definitions
    # Pelvis (calculated)
    h36m_joints[0] = (keypoints[11] + keypoints[12]) / 2 
    # Right Leg
    h36m_joints[1], h36m_joints[2], h36m_joints[3] = keypoints[12], keypoints[14], keypoints[16]
    # Left Leg
    h36m_joints[4], h36m_joints[5], h36m_joints[6] = keypoints[11], keypoints[13], keypoints[15]
    # Spine and Head (calculated)
    h36m_joints[8] = (keypoints[5] + keypoints[6]) / 2   # Thorax
    h36m_joints[7] = (h36m_joints[0] + h36m_joints[8]) / 2 # Spine
    h36m_joints[9] = (h36m_joints[8] * 0.75) + (keypoints[0] * 0.25) # Neck
    h36m_joints[10] = keypoints[0] + (keypoints[0] - h36m_joints[9]) # Head
    # Arms
    h36m_joints[11], h36m_joints[12], h36m_joints[13] = keypoints[5], keypoints[7], keypoints[9]
    h36m_joints[14], h36m_joints[15], h36m_joints[16] = keypoints[6], keypoints[8], keypoints[10]
    
    return h36m_joints
# In vis.py, replace the draw_3d_skeleton function

def draw_3d_skeleton(pose_3d, skeleton_parents, ax):
    """
    Renders a single 3D skeleton, scales it, rotates it 90 degrees clockwise,
    and orients it correctly for visualization.
    """
    pose_3d = pose_3d.copy()
    
    # Scale the pose to be 4x larger (3000mm)
    pose_3d *= 3

    # 2. Center the pose at the origin before rotating
    # The pelvis (joint 0) is our center of rotation
    pelvis_point = pose_3d[0].copy()
    pose_3d -= pelvis_point

    # 3. Apply a 90-degree clockwise rotation around the vertical (Z) axis
    # The rotation matrix for -90 degrees around Z is [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    rotated_pose = np.zeros_like(pose_3d)
    rotated_pose[:, 0] = - pose_3d[:, 0]   
    rotated_pose[:, 1] = -pose_3d[:, 2]  # New Y is the negative old X
    rotated_pose[:, 2] =  pose_3d[:, 1]   # Z stays the same
    
    # 4. Translate the rotated pose back to its original position
    pose_3d = rotated_pose + pelvis_point
    
    # 5. Flip the skeleton upright and place it on the ground
    pose_3d[:, 2] *= -1 # Invert the Z-axis
    pose_3d[:, 2] -= np.min(pose_3d[:, 2]) # Ground the skeleton at z=0

    ax.clear()
    ax.view_init(elev=15., azim=70)
    
    # 6. Update the plot radius to fit the larger skeleton
    radius = 3000
    ax.set_xlim3d([-radius, radius]); ax.set_ylim3d([-radius, radius]); ax.set_zlim3d([0, radius * 1.5])
    ax.set_xlabel("X"); ax.set_ylabel("Y (Depth)"); ax.set_zlabel("Z (Height)")
    ax.set_title("3D Pose Reconstruction")
    left_joints  = {4, 5, 6, 11, 12, 13}   # Left leg + left arm
    right_joints = {1, 2, 3, 14, 15, 16}   # Right leg + right arm
    # Draw the skeleton bones and joints
    bones = [(p, c) for c, p in enumerate(skeleton_parents) if p != -1]
    for parent, child in bones:
        if child in left_joints or parent in left_joints:
            color = "blue"
        elif child in right_joints or parent in right_joints:
            color = "red"
        else:
            color = "black"  # torso/center bones
        ax.plot(
            [pose_3d[parent, 0], pose_3d[child, 0]],
            [pose_3d[parent, 1], pose_3d[child, 1]],
            [pose_3d[parent, 2], pose_3d[child, 2]],
            zdir="z", color=color, linewidth=3
        )
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], zdir='z', c='black', s=20)

def fig_to_array(fig):
    """Converts a matplotlib figure to a numpy array (image)."""
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# --- Main Application Logic ---
def main(cli_args):
    receptive_field = cli_args.number_of_frames

    # --- Load Your Trained 3D Model (SST_Model) ---
    print("Loading 3D SST_Model...")
    sst_model = SST_Model(cli_args, num_joints=17, in_chans=4)
    if torch.cuda.is_available(): sst_model = torch.nn.DataParallel(sst_model).cuda()
    checkpoint = torch.load(cli_args.checkpoint_3d, map_location=lambda storage, loc: storage)
    sst_model.load_state_dict(checkpoint['model_pos'])
    sst_model.eval()

    # --- Load Your Pre-trained 2D Model (YOLOv8-Pose) ---
    print("Loading 2D YOLOv8-Pose model...")
    pose_model = YOLO('yolov8n-pose.pt') 

    # --- Initialize Video I/O ---
    cap = cv2.VideoCapture(cli_args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output path automatically
    video_name = os.path.basename(cli_args.video)
    video_name_no_ext = os.path.splitext(video_name)[0]
    output_dir = 'output/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_video_path = os.path.join(output_dir, f"{video_name_no_ext}_3d.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))

    # --- NEW: Create buffers for both keypoints and original video frames ---
    keypoints_buffer = collections.deque(maxlen=receptive_field)
    frames_buffer = collections.deque(maxlen=receptive_field)
    # --------------------------------------------------------------------
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get skeleton definition from the H36M dataset
    dataset = Human36mDataset('data/data_3d_h36m.npz')
    skeleton_parents = dataset.skeleton().parents()

    # Setup the 3D plot
    fig = plt.figure(figsize=(width/100, height/100))
    ax = fig.add_subplot(111, projection='3d')

    # --- Main Processing Loop ---
    for frame_idx in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret: break

        # Run YOLOv8-Pose to get 2D Poses
        results = pose_model(frame, verbose=False)
        if len(results) > 0 and results[0].keypoints.shape[0] > 0:
            keypoints_coco = results[0].keypoints.xy[0].cpu().numpy()
            keypoints_h36m = coco_to_h36m(keypoints_coco)
        else:
            keypoints_h36m = np.zeros((17, 2), dtype=np.float32)

        # --- NEW: Add both the keypoints and the original frame to their buffers ---
        keypoints_buffer.append(keypoints_h36m)
        frames_buffer.append(frame)
        # -------------------------------------------------------------------------
            
        # Once the buffers are full, start predicting and writing to the output video
        if len(keypoints_buffer) == receptive_field:
            # --- NEW: Get the correctly synchronized frame from the buffer ---
            # The model predicts the pose for the center frame of the sequence.
            # So, we need to retrieve the original video frame from that same moment in time.
            center_frame_index = receptive_field // 2
            synced_frame = frames_buffer[center_frame_index]
            synced_keypoints = keypoints_buffer[center_frame_index]
            # -------------------------------------------------------------

            # Draw the 2D skeleton on the synchronized input frame
            for joint in synced_keypoints: 
                cv2.circle(synced_frame, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1)
            
            # Prepare the full sequence for the 3D model
            sequence_2d = np.array(keypoints_buffer, dtype=np.float32)
            sequence_2d_normalized = normalize_screen_coordinates(sequence_2d, width, height)
            sequence_4d = add_motion_dynamics(sequence_2d_normalized)
            input_tensor = torch.from_numpy(sequence_4d).float().unsqueeze(0).cuda()

            with torch.no_grad():
                predicted_3d_sequence = sst_model(input_tensor)
            
            predicted_3d_pose = predicted_3d_sequence[0, center_frame_index].cpu().numpy()
            predicted_3d_pose *= 1000 # Scale up to millimeters
            
            # Render the 3D skeleton to an image
            draw_3d_skeleton(predicted_3d_pose, skeleton_parents, ax)
            skeleton_img = fig_to_array(fig)
        
            # Create a side-by-side view and save it to the output video
            combined_frame = np.hstack((synced_frame, skeleton_img))
            out_video.write(combined_frame)

    # --- Cleanup ---
    cap.release(); out_video.release(); plt.close(fig)
    print(f"Processing finished. Output video saved to {output_video_path}")

if __name__ == '__main__':
    # Use the main argument parser to get model parameters like --frame, --depth, etc.
    args = parse_args()
    
    # Create a temporary parser for video-specific arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--checkpoint-3d', type=str, required=True, help='Path to the trained 3D model checkpoint')
    vis_args, _ = parser.parse_known_args()
    
    # Add the video-specific arguments to the main args object
    args.video = vis_args.video
    args.checkpoint_3d = vis_args.checkpoint_3d

    main(args)

