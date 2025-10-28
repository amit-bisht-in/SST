# In common/model_sst.py

# --- Library Imports ---
import math  # Used for math operations
from functools import partial  # Used to 'freeze' arguments for a function
import torch  # The main PyTorch library for neural networks
import torch_dct as dct # A library for the Discrete Cosine Transform (used in FreqMlp)
import torch.nn as nn  # The building blocks for neural networks (Layers, Models, etc.)
from timm.models.layers import DropPath  # A special type of dropout for regularization
from einops import rearrange  # A library to make reshaping tensors (multi-dimensional arrays) easier

# --- Model Building Blocks ---
# We define the smaller "Lego bricks" of our model first.

class Mlp(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) block, also known as a Feed-Forward Network.
    It's just two "Linear" (fully-connected) layers with an activation function in between.
    Its job is to process and transform the features at a specific step.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # This function sets up the layers. It runs once when the model is created.
        super().__init__()
        
        # If the user doesn't specify an output size, make it the same as the input size
        out_features = out_features or in_features
        # If the user doesn't specify a hidden size, make it the same as the input size
        hidden_features = hidden_features or in_features
        
        # Define the first linear layer (e.g., takes 512 features in, outputs 2048)
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Define the activation function (GELU is a smooth version of the common ReLU)
        self.act = act_layer()
        # Define the second linear layer (e.g., takes 2048 features in, outputs 512)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Define a dropout layer, which randomly turns off neurons during training to prevent overfitting
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # This 'forward' function defines what happens when data flows *through* the block.
        x = self.fc1(x)     # Pass data through the first layer
        x = self.act(x)      # Apply the activation function
        x = self.drop(x)     # Apply dropout
        x = self.fc2(x)     # Pass data through the second layer
        x = self.drop(x)     # Apply dropout again
        return x             # Return the final processed data

class FreqMlp(nn.Module):
    """
    A special MLP that operates in the Frequency Domain.
    This block is designed to learn patterns in *motion* (how things change over time)
    instead of just looking at static positions.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # This setup is identical to the standard Mlp
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # The input 'x' has a shape like (batch_size, num_frames, num_features)
        b, f, _ = x.shape
        
        # 1. Transform from Time Domain to Frequency Domain
        #    dct() calculates the Discrete Cosine Transform.
        #    This converts the sequence of poses (time) into a set of frequencies (motion patterns).
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        # 2. Process the data with the MLP *while it's in the frequency domain*
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        
        # 3. Transform back from Frequency Domain to Time Domain
        #    idct() is the Inverse Discrete Cosine Transform.
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x

class Attention(nn.Module):
    """
    Standard Multi-Head Self-Attention. This is the heart of the Transformer.
    It allows each frame in the sequence to "look at" and share information
    with every other frame, learning the relationships between them.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # e.g., 8 "attention heads"
        head_dim = dim // num_heads # The dimension for each individual head
        # This is a scaling factor to keep the math stable during training
        self.scale = qk_scale or head_dim ** -0.5
        
        # A single, large linear layer that will create the Query, Key, and Value
        # for all heads at once.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) # Dropout for the attention scores
        
        # A final linear layer to combine the results from all heads
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) # Dropout for the output
        
    def forward(self, x):
        B, N, C = x.shape # Batch Size, Sequence Length (Frames), Channels (Embedding Dim)
        
        # 1. Project input to Query (Q), Key (K), and Value (V) for all heads
        #    B, N, C -> B, N, 3 (q,k,v), num_heads, head_dim -> 3, B, num_heads, N, head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Separate Q, K, and V
        
        # 2. Calculate Attention Scores
        #    (Q @ K.transpose) creates a "similarity matrix": how much each frame (Q)
        #    matches every other frame (K). We then scale it.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        #    softmax() turns these scores into probabilities (all rows add up to 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 3. Apply Attention to Values
        #    The scores (attn) are used to create a weighted sum of the Values (v).
        #    This results in a new sequence where each frame is a mix of information
        #    from all other frames, based on the attention scores.
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 4. Project the combined output back to the original dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    A standard Transformer Encoder block.
    It's made of two main parts: an Attention block and an MLP block.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim) # Normalization layer before attention
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        # DropPath is a regularization technique that randomly skips a whole block
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) # Normalization layer before the MLP
        mlp_hidden_dim = int(dim * mlp_ratio) # The MLP's hidden layer is often 4x bigger
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # This is the "pre-normalization" style found in many Transformers
        # 1. Apply attention, then add the original input (this is a "residual connection")
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 2. Apply the MLP, then add the result from step 1 (another residual connection)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MixedBlock(nn.Module):
    """
    Our custom Transformer block with Dynamic Gating.
    It processes features with both a standard MLP (for spatial features) and a Frequency MLP (for motion).
    It uses a learned "gate" to decide the best way to combine them.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, gate_mode='dynamic'):
        super().__init__()
        self.gate_mode = gate_mode # Store the mode ('dynamic', 'spatial_only', etc.)
        
        # Standard Attention Block (the first part of the Transformer block)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Define the two "expert" MLPs for the second part
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) # Spatial MLP
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) # Frequency MLP

        # --- THIS IS THE FIX ---
        # Only create the gate network if we are in 'dynamic' mode
        if self.gate_mode == 'dynamic':
            self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        # --------------------

    def forward(self, x):
        # 1. The first part is always the same: apply attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # 2. Normalize the output from the attention block
        norm_x = self.norm2(x)

        # --- THIS IS THE FIX ---
        # Re-add the full logic to switch between modes
        if self.gate_mode == 'spatial_only':
            # Force 100% spatial: only use the standard MLP
            mlp_output = self.mlp1(norm_x)
            
        elif self.gate_mode == 'temporal_only':
            # Force 100% temporal: only use the Frequency MLP
            mlp_output = self.mlp2(norm_x)
            
        elif self.gate_mode == 'additive':
            # Additive Fusion: Use 100% of both and add them
            spatial_output = self.mlp1(norm_x)
            temporal_output = self.mlp2(norm_x)
            mlp_output = spatial_output + temporal_output
            
        else: # Default 'dynamic' mode
            # Use the learned gate to find the best mix
            gate_values = self.gate(norm_x) # Calculate weights (e.g., 0.7)
            # Combine the two MLPs. e.g., 0.7 * spatial + (1 - 0.7) * temporal
            mlp_output = gate_values * self.mlp1(norm_x) + (1 - gate_values) * self.mlp2(norm_x)
        # --------------------
            
        # 4. Apply the final residual connection
        x = x + self.drop_path(mlp_output)
            
        return x

class SST_Model(nn.Module):
    """
    The main model class that assembles the full architecture.
    SST: Spatial-Spectral Transformer
    """
    def __init__(self, opt, num_joints=17, in_chans=4, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, norm_layer=None):
        # This function sets up all the layers of the model based on the command-line arguments ('opt')
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # --- Read parameters from the 'opt' (args) object ---
        # Dropout rates are now read from args, activating them
        drop_rate = opt.dropout if hasattr(opt, 'dropout') else 0.1
        attn_drop_rate = opt.dropout if hasattr(opt, 'dropout') else 0.1
        drop_path_rate = opt.dropout if hasattr(opt, 'dropout') else 0.1
        
        depth = opt.depth # Number of Transformer blocks
        embed_dim_ratio = opt.embed_dim_ratio if hasattr(opt, 'embed_dim_ratio') else 32
        
        # Calculate the main embedding dimension (the "width" of the Transformer)
        embed_dim = embed_dim_ratio * num_joints # e.g., 32 * 17 = 544
        # Calculate the final output dimension
        out_dim = num_joints * 3 # e.g., 17 joints * 3 (x,y,z) coordinates = 51
        
        # Get the frame and coefficient counts from the command line
        self.num_frame_kept = opt.number_of_kept_frames
        self.num_coeff_kept = opt.number_of_kept_coeffs if hasattr(opt, 'number_of_kept_coeffs') and opt.number_of_kept_coeffs is not None else self.num_frame_kept

        # --- Define all the Model Layers ---
        
        # 1. CNN Stem: A small 1D convolution network to process the 4-channel input (x, y, vx, vy)
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1)
        )

        # 2. Embedding Layers: These project the input data into the high-dimensional space
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)
        
        # 3. Positional Embeddings: These are learnable "vectors" that are added to the
        #    input tokens to tell the model *which* joint or *which* frame it is looking at.
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 4. Stochastic Depth: A regularization technique for deep networks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 5. Spatial Transformer Blocks
        #    This is a stack of standard Transformer blocks that process the spatial information
        #    (the relationship between different joints in a single frame).
        self.Spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        # --- THIS IS THE FIX ---
        # 6. Mixed (Spatio-Temporal) Transformer Blocks
        #    This is a stack of our custom MixedBlocks that process the combined data
        #    (both spatial and frequency information).
        #    We read the gate_mode from the command-line args.
        gate_mode = opt.gate_mode if hasattr(opt, 'gate_mode') else 'dynamic'
        self.blocks = nn.ModuleList([
            MixedBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                       gate_mode=gate_mode) # <-- Pass the mode here
            for i in range(depth)])
        # --------------------
            
        # 7. Normalization Layers
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        
        # 8. Output Head: A simple MLP to project the final features to 3D coordinates
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))

    def Spatial_forward_features(self, x):
        """This helper function runs the data through the 'Spatial' stream."""
        b, f, p, _ = x.shape
        num_frame_kept = self.num_frame_kept
        
        # Select the frames to keep (e.g., 81 frames) from the input (e.g., 243 frames)
        index = torch.arange((f - 1) // 2 - num_frame_kept // 2, (f - 1) // 2 + num_frame_kept // 2 + 1)
        
        # Embed the joints and add the spatial positional embedding
        x = self.Joint_embedding(x[:, index].reshape(b * num_frame_kept, p, -1))
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        
        # Pass through the spatial-only Transformer blocks
        for blk in self.Spatial_blocks:
            x = blk(x)
            
        x = self.Spatial_norm(x)
        # Reshape the output to be (batch, frames, features)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x

    def forward_features(self, x, Spatial_feature):
        """This helper function runs the data through the 'Temporal' (Frequency) stream."""
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept
        
        # Apply DCT to get frequency coefficients (the motion patterns)
        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        
        # Embed the frequencies and add the temporal positional embedding
        x = self.Freq_embedding(x)
        Spatial_feature += self.Temporal_pos_embed
        x += self.Temporal_pos_embed_
        
        # Concatenate the spatial and temporal features to create the final sequence
        x = torch.cat((x, Spatial_feature), dim=1)
        
        # Pass the combined sequence through our MixedBlocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        """
        This is the main function that defines the complete flow of data through the model.
        """
        
        # Fix the input shape if it's in a strange format
        if x.shape[-1] == 1 and len(x.shape) == 5:
            x = x.squeeze(-1)
        
        b, f, p, c = x.shape # (Batch, Frames, Joints, Channels)
        
        # 1. Apply the CNN Stem to the input
        #    We reshape to (Batch*Frames, Channels, Joints) to use Conv1d
        x_reshaped = x.view(b * f, p, c).permute(0, 2, 1)
        x_processed = self.cnn_stem(x_reshaped)
        #    Reshape back to (Batch, Frames, Joints, Channels)
        x = x_processed.permute(0, 2, 1).contiguous().view(b, f, p, c)
        
        # 2. Run the two parallel streams
        x_ = x.clone() # Create a copy of the input
        Spatial_feature = self.Spatial_forward_features(x) # Run the Spatial stream
        x = self.forward_features(x_, Spatial_feature)     # Run the Temporal/Mixed stream
        
        # 3. Get the final output tokens
        #    The output 'x' is a sequence of (Frequency_Tokens, Spatial_Tokens).
        #    We only care about the Spatial_Tokens for our final 3D pose prediction.
        x_frame_tokens = x[:, self.num_coeff_kept:]
        
        # 4. Apply the output Head to predict the 3D coordinates
        pred_poses = self.head(x_frame_tokens)
        
        # 5. Reshape to our desired output: (batch, frames, joints, 3)
        pred_poses = pred_poses.view(b, self.num_frame_kept, p, 3)
        
        return pred_poses

