# In common/visual.py

import torch
import numpy as np
import cv2
import collections
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse # <-- NEW IMPORT

# --- Project Imports ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.camera import normalize_screen_coordinates
from common.model_sst import SST_Model
from common.h36m_dataset import Human36mDataset

# --- Ultralytics Import for 2D Pose Detection ---
from ultralytics import YOLO



# In common/visual.py
# In common/visual.py, replace the coco_to_h36m function
# In common/visual.py, replace the coco_to_h36m function
# In common/visual.py

def coco_to_h36m(keypoints):
    """
    Converts 17-joint COCO format keypoints to the 17-joint H36M format.
    This is the corrected version that fixes the head and limb alignment.
    """
    # Create a new array for H36M keypoints
    h36m_joints = np.zeros((17, 2), dtype=np.float32)

    # --- JOINT MAPPING ---
    # This maps the 17 COCO joints to the 17 H36M joints
    # The H36M skeleton definition used by your model is:
    # 0: Pelvis, 1: R_Hip, 2: R_Knee, 3: R_Ankle, 4: L_Hip, 5: L_Knee, 6: L_Ankle, 
    # 7: Spine, 8: Thorax, 9: Neck, 10: Head, 11: L_Shoulder, 12: L_Elbow, 
    # 13: L_Wrist, 14: R_Shoulder, 15: R_Elbow, 16: R_Wrist

    # Direct mappings from COCO indices to H36M indices
    h36m_joints[1] = keypoints[12]  # R_Hip
    h36m_joints[2] = keypoints[14]  # R_Knee
    h36m_joints[3] = keypoints[16]  # R_Ankle
    h36m_joints[4] = keypoints[11]  # L_Hip
    h36m_joints[5] = keypoints[13]  # L_Knee
    h36m_joints[6] = keypoints[15]  # L_Ankle
    
    h36m_joints[11] = keypoints[5]   # L_Shoulder
    h36m_joints[12] = keypoints[7]   # L_Elbow
    h36m_joints[13] = keypoints[9]   # L_Wrist
    h36m_joints[14] = keypoints[6]   # R_Shoulder
    h36m_joints[15] = keypoints[8]   # R_Elbow
    h36m_joints[16] = keypoints[10]  # R_Wrist

    # Calculated joints for H36M
    h36m_joints[0] = (keypoints[11] + keypoints[12]) / 2  # Pelvis (midpoint of hips)
    h36m_joints[8] = (keypoints[5] + keypoints[6]) / 2   # Thorax (midpoint of shoulders)
    h36m_joints[7] = (h36m_joints[0] + h36m_joints[8]) / 2 # Spine (midpoint of pelvis and thorax)
    
    # A better approximation for Neck and Head
    h36m_joints[9] = (keypoints[0] + h36m_joints[8]) / 2      # Neck (midpoint of nose and thorax)
    h36m_joints[10] = keypoints[0]                             # Head (use nose as the head top)

    return h36m_joints


# --- Helper Functions (add_motion_dynamics, draw_3d_skeleton, fig_to_array) ---
# ... (These functions are the same as before) ...
def add_motion_dynamics(sequence_2d):
    velocity = np.diff(sequence_2d, axis=0, prepend=sequence_2d[0:1])
    return np.concatenate((sequence_2d, velocity), axis=-1)




def draw_3d_skeleton(pose_3d, skeleton_parents, ax):

    pose_3d = pose_3d.copy() # Make a copy to avoid changing the original data
    pose_3d[:, 2] *= -1 
    """Draws a single 3D skeleton on a matplotlib Axes3D object with ground and grid."""
    ax.clear()
    
    # 1. Set view point: Flip the Y-axis for correct orientation (Y-up)
    # The default matplotlib Y-axis points down (or into the screen). 
    # By setting elev and azim, we control the viewing angle.
    # We also flip the Y-axis to make it point upwards in the plot.
    ax.view_init(elev=15., azim=70) # Standard viewing angle
    ax.set_box_aspect([1, 1, 1]) # Equal aspect ratio for all axes
    
    # 2. Set Axis Limits and Labels
    # We'll use a fixed range around the pelvis (joint 0) for consistent scaling
    # Adjust these values based on your expected human size in mm
    pelvis_x, pelvis_y, pelvis_z = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    
    # Define the range for the plot to encompass a person
    # You might need to adjust these limits based on your data's actual range
    plot_range = 1000 # +/- 1 meter from the center for X, Y, Z for better visibility
    
    ax.set_xlim3d([pelvis_x - plot_range/2, pelvis_x + plot_range/2])
    ax.set_ylim3d([pelvis_y - plot_range/2, pelvis_y + plot_range/2])
    # Z-axis (height) usually starts near 0 for the ground
    ax.set_zlim3d([0, pelvis_z + plot_range]) # Assuming pelvis_z is a good lower bound, extend upwards

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    
    # 3. Add a Ground Plane (a simple grid)
    # The ground will be at z=0. We'll draw lines to form a grid.
    grid_step = 200 # 200 mm between grid lines
    
    # Get current plot limits for the ground
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()
    
    # Draw lines parallel to X-axis
    for y_coord in np.arange(y_min, y_max, grid_step):
        ax.plot([x_min, x_max], [y_coord, y_coord], [0, 0], color='gray', linestyle='--', linewidth=0.5)
    # Draw lines parallel to Y-axis
    for x_coord in np.arange(x_min, x_max, grid_step):
        ax.plot([x_coord, x_coord], [y_min, y_max], [0, 0], color='gray', linestyle='--', linewidth=0.5)

    # 4. Draw the Skeleton
    bones = [(p, c) for c, p in enumerate(skeleton_parents) if p != -1]
    for parent, child in bones:
        ax.plot([pose_3d[parent, 0], pose_3d[child, 0]],
                [pose_3d[parent, 1], pose_3d[child, 1]],
                [pose_3d[parent, 2], pose_3d[child, 2]],
                zdir='z', color='dodgerblue', linewidth=3)
    
    # Draw joints as points
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], zdir='z', c='red', s=20)
    
    ax.set_title("3D Pose Reconstruction", pad=0)
    
    # Turn off axes labels if they clutter, but keep grid for scale
    # ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.grid(True) # Ensure the grid lines are visible



def fig_to_array(fig):
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


# --- Main Application Logic ---
def main(args):
    receptive_field = 27 # The model expects a 27-frame sequence

    # --- Load Your Trained 3D Model (SST_Model) ---
    print("Loading 3D SST_Model...")
    # Create a dummy args object for the model, as it expects some parameters
    # Replace with this line
    model_args = type('Args', (object,), {'number_of_frames': receptive_field, 'number_of_kept_frames': receptive_field, 
                                        'number_of_kept_coeffs': receptive_field, 'depth': 4, 'dropout': 0.1, 
                                        'embed_dim_ratio': 32, 'n_heads': 8})()
    sst_model = SST_Model(model_args, num_joints=17, in_chans=4)
    if torch.cuda.is_available():
        sst_model = torch.nn.DataParallel(sst_model).cuda()
    checkpoint = torch.load(args.checkpoint_3d, map_location=lambda storage, loc: storage)
    sst_model.load_state_dict(checkpoint['model_pos'])
    sst_model.eval()

    # --- Load Your Pre-trained 2D Model (YOLOv8-Pose) ---
    print("Loading 2D YOLOv8-Pose model...")
    pose_model = YOLO('yolov8n-pose.pt') 

    # --- 2. INITIALIZE VIDEO I/O AND DATA ---
    print(f"Opening video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.video}")
    
    # --- NEW: Automatically create the output path ---
    video_name = os.path.basename(args.video)
    video_name_no_ext = os.path.splitext(video_name)[0]
    output_video_path = os.path.join(args.output_dir, f"{video_name_no_ext}_sst.mp4")
    # ----------------------------------------------------
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)); fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))

    keypoints_buffer = collections.deque(maxlen=receptive_field)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    dataset = Human36mDataset('data/data_3d_h36m.npz')
    skeleton_parents = dataset.skeleton().parents()

    fig = plt.figure(figsize=(width/100, height/100))
    ax = fig.add_subplot(111, projection='3d')

    # --- 3. MAIN PROCESSING LOOP ---
    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret: break

        results = pose_model(frame, verbose=False)
        if results[0].keypoints.shape[0] > 0:
            keypoints_2d = results[0].keypoints.xy[0].cpu().numpy()
        else:
            keypoints_2d = np.zeros((17, 2), dtype=np.float32)

        keypoints_buffer.append(keypoints_2d)
        
        for joint in keypoints_2d: cv2.circle(frame, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1)
            
        skeleton_img = np.zeros_like(frame)

        if len(keypoints_buffer) == receptive_field:
            sequence_2d = np.array(keypoints_buffer, dtype=np.float32)
            sequence_2d_normalized = normalize_screen_coordinates(sequence_2d, width, height)
            sequence_4d = add_motion_dynamics(sequence_2d_normalized)
    

            input_tensor = torch.from_numpy(sequence_4d).float().unsqueeze(0).cuda()

         

            # --- Wrap the prediction in torch.no_grad() ---
            with torch.no_grad():
                predicted_3d_sequence = sst_model(input_tensor)
            # ---------------------------------------------

            
            predicted_3d_pose = predicted_3d_sequence[0, receptive_field // 2].cpu().numpy()


           
            predicted_3d_pose *= 1000 # Scale from meters/normalized to millimeters
            
            draw_3d_skeleton(predicted_3d_pose, skeleton_parents, ax)
            skeleton_img = fig_to_array(fig)
        
        combined_frame = np.hstack((frame, skeleton_img))
        out_video.write(combined_frame)

    # --- 4. CLEANUP ---
    cap.release(); out_video.release(); plt.close(fig)
    print(f"Processing finished. Output video saved to {output_video_path}")

if __name__ == '__main__':
    # --- NEW: Argument Parser for the standalone script ---
    parser = argparse.ArgumentParser(description='SST Model Video Demo')
    parser.add_argument('--checkpoint-3d', type=str, default='checkpoint/SST_Model_final/best_epoch.bin', help='Path to the trained 3D model checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output-dir', type=str, default='output/', help='Directory to save the output video')
    cli_args = parser.parse_args()
    
    main(cli_args)