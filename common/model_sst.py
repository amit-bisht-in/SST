## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from einops import rearrange, repeat

import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import DropPath

# torch.autograd.set_detect_anomaly(True)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.attn2 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        y_ = self.attn1(self.norm1(y))
        y = y + self.drop_path(y_)
        y = y + self.drop_path(self.mlp1(self.norm2(y)))

        x = x + self.drop_path(self.attn2(self.norm3(x)))
        x = x + self.cross_attn(x, y_)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        return x, y


# Replace the old MixedBlock with this one

class MixedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # --- This is the new gating mechanism ---
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # --- This is the new gated MLP logic ---
        norm_x = self.norm2(x)
        gate_values = self.gate(norm_x)
        gated_mlp_output = gate_values * self.mlp1(norm_x) + (1 - gate_values) * self.mlp2(norm_x)
        x = x + self.drop_path(gated_mlp_output)
        
        return x
    


# Replace with this line
class SST_Model(nn.Module):
    
    # In model_sst.py, inside the SST_Model class

    def __init__(self, opt, num_frame=9, num_joints=17, in_chans=4, # MODIFIED: Default channels is now 4
                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  norm_layer=None, args=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Parameters from opt object and defaults
        embed_dim_ratio = 32
        depth = opt.depth
        drop_path_rate = opt.dropout if hasattr(opt, 'dropout') else drop_path_rate
        
        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3
        self.num_frame_kept = opt.number_of_kept_frames
        self.num_coeff_kept = opt.number_of_kept_coeffs if opt.number_of_kept_coeffs is not None else self.num_frame_kept

        # --- NEW: Skeleton definition for bone loss ---
      
        # --- NEW: Convolutional Stem for local feature extraction ---
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1)
        )

        # Embeddings for joints and frequency coefficients
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)
        
        # Positional embeddings
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # Use the Gated MixedBlock
        if hasattr(opt, 'naive') and opt.naive:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                MixedBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
            
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # --- MODIFIED: Simplified Head for multi-frame output ---
        # The old aggregation layers and concatenated head are removed.
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )



    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        num_frame_kept = self.num_frame_kept

        index = torch.arange((f-1)//2-num_frame_kept//2, (f-1)//2+num_frame_kept//2+1)

        x = self.Joint_embedding(x[:, index].view(b*num_frame_kept, p, -1))
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x

    def forward_features(self, x, Spatial_feature):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept

        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        x = self.Freq_embedding(x) 
        
        Spatial_feature += self.Temporal_pos_embed
        x += self.Temporal_pos_embed_
        x = torch.cat((x, Spatial_feature), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        return x

   # Replace the forward method in your SST_Model class with this one

    def forward(self, x):
        # The original repo's input can have a strange shape, so we fix it first
        if x.shape[-1] == 1 and len(x.shape) == 5:
            x = x.squeeze(-1)
        if len(x.shape) == 4 and x.shape[-1] != 4: # If input is (B, F, J, C) but C is not 4
            x = x.permute(0, 3, 1, 2) # Assuming (B, C, F, J) -> (B, F, J, C)
            x = x.permute(0, 2, 3, 1)

        b, f, p, c = x.shape
        
        # --- NEW: Apply the CNN Stem ---
        x_reshaped = x.view(b * f, p, c).permute(0, 2, 1)
        x_processed = self.cnn_stem(x_reshaped)
        x = x_processed.permute(0, 2, 1).contiguous().view(b, f, p, c)
        
        # Run the main Transformer blocks
        x_ = x.clone()
        Spatial_feature = self.Spatial_forward_features(x)
        x = self.forward_features(x_, Spatial_feature)
        
        # --- NEW: Multi-frame Prediction Logic ---
        # Select only the frame-related tokens for the output
        x_frame_tokens = x[:, self.num_coeff_kept:]
        
        # Apply the head to each token in the sequence
        pred_poses = self.head(x_frame_tokens)
        
        # Reshape to our desired output: (batch, frames, joints, 3)
        pred_poses = pred_poses.view(b, self.num_frame_kept, p, 3)
        
        return pred_poses

