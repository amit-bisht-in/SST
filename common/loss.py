# In common/loss.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# --- Original Metric Functions (Unchanged) ---

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

# In common/loss.py, replace the n_mpjpe function

def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only).
    """
    assert predicted.shape == target.shape

    # This version is corrected to work with 3D tensors of shape (N, J, 3)
    # The xyz dimension is 2, and the joints dimension is 1.
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=2, keepdim=True), dim=1, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=2, keepdim=True), dim=1, keepdim=True)

    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


# In common/loss.py, replace the p_mpjpe function

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # This block correctly handles reflections on a batch-wise basis
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Recompute rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation

    predicted_aligned = a*np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))




def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

# --- Upgraded Loss Functions for Our Model ---

def calculate_bone_lengths(pose_3d, bones):
    """Calculates bone lengths from a 3D pose tensor."""
    # pose_3d shape: (batch, seq_len, num_joints, 3)
    j1_indices = [b[0] for b in bones]
    j2_indices = [b[1] for b in bones]
    
    # Gather the 3D coordinates of the start and end joints for each bone
    j1_coords = pose_3d[:, :, j1_indices, :]
    j2_coords = pose_3d[:, :, j2_indices, :]
    
    # Calculate the Euclidean distance (L2 norm) between the joint pairs
    return torch.norm(j1_coords - j2_coords, dim=-1)

def compute_total_loss(pred_sequence_3d, gt_sequence_3d, bones, lambda_smooth=1.0, lambda_bone=0.5):
    """
    Computes a combined loss of pose accuracy, motion smoothness, and bone length consistency.
    """
    # 1. Pose Loss (on the full sequence)
    # This now correctly compares two sequences of the same shape
    pose_loss = mpjpe(pred_sequence_3d, gt_sequence_3d)

    # 2. Temporal Smoothing Loss (on the full predicted sequence)
    velocity = pred_sequence_3d[:, 1:] - pred_sequence_3d[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    smooth_loss = torch.mean(torch.norm(acceleration, dim=-1))
    
    # 3. Bone Length Consistency Loss (on the full predicted sequence)
    bone_lengths = calculate_bone_lengths(pred_sequence_3d, bones)
    max_lengths, _ = torch.max(bone_lengths, dim=1, keepdim=True)
    bone_loss = F.mse_loss(bone_lengths, max_lengths.expand_as(bone_lengths))
    
    # 4. Combine the losses
    total_loss = pose_loss + lambda_smooth * smooth_loss + lambda_bone * bone_loss
    
    loss_dict = { "total_loss": total_loss, "pose_loss": pose_loss, "smooth_loss": smooth_loss, "bone_loss": bone_loss }
    return total_loss, loss_dict