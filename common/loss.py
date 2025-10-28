# In common/loss.py

# Import the necessary libraries
import torch # The main PyTorch library
import torch.nn as nn # Neural network modules from PyTorch (like loss functions)
import numpy as np # A library for numerical operations
import torch.nn.functional as F # A part of PyTorch with more functions

# --- Evaluation Metrics ---
# These functions are used to measure the model's performance during evaluation.
# (mpjpe, n_mpjpe, p_mpjpe, mean_velocity_error are all unchanged)

def mpjpe(predicted, target):
    """
    Mean Per-Joint Position Error (MPJPE).
    This is the most common and important metric (Protocol #1).
    It measures the average distance in 3D space between each predicted joint
    and its correct, ground-truth position.
    """
    # Check that the prediction and target tensors have the exact same shape
    assert predicted.shape == target.shape
    
    # torch.norm calculates the Euclidean distance (sqrt(x^2 + y^2 + z^2))
    # for each joint.
    # dim=len(target.shape)-1 tells it to calculate this along the last dimension (the 3 coordinates).
    # torch.mean then calculates the average error across all joints and all frames.
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))



def compute_total_loss(pred_sequence_3d, gt_sequence_3d, lambda_jerk=0.001):
    """
    Computes a combined loss of pose accuracy and a *weighted temporal jerk* (smoothness).
    This penalizes jittery motion (high jerk) more than fast, smooth motion.
    """
    
    # 1. Pose Loss (Accuracy)
    # How far is the prediction from the correct answer?
    pose_loss = mpjpe(pred_sequence_3d, gt_sequence_3d)

    return pose_loss

