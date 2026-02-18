import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any, List, Optional
from .BaseModel import BaseModel

class TemporalSwinModel(BaseModel):
    """
    Level 3: Temporal + Swin.
    
    Upgrades the Dynamic and State encoders to use 3D Convolutions.
    This allows the model to learn temporal trends (e.g., wind changes, fire spread velocity)
    instead of just averaging the history.
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str = "Dice",
        use_doy: bool = False,
        required_img_size: Optional[List[int]] = None,
        hidden_dim: int = 96,
        n_leading_observations: int = 5, # New arg to define kernel size T
        *args: Any,
        **kwargs: Any
    ):
        if required_img_size is None:
            required_img_size = [128, 128]

        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=False, 
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=use_doy,
            required_img_size=required_img_size,
            *args,
            **kwargs
        )
        self.save_hyperparameters()

        # --- Feature Definitions ---
        self.static_indices = [12, 13, 14] + list(range(16, 33))
        self.dynamic_indices = list(range(12)) + [15] + list(range(33, 39))
        self.fire_index = [39]

        n_static = len(self.static_indices)
        n_dynamic = len(self.dynamic_indices)
        n_fire = len(self.fire_index)
        
        # Time dimension (T)
        T = n_leading_observations

        # --- Encoders ---
        
        # 1. Static Encoder (No time dimension here, simple 2D)
        self.static_encoder = nn.Sequential(
            nn.Conv2d(n_static, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # 2. Dynamic Encoder (3D Conv)
        # Input: (B, C, T, H, W)
        # Kernel: (T, 3, 3) -> Collapses Time to 1, Spatial processing 3x3
        self.dynamic_encoder = nn.Sequential(
            nn.Conv3d(n_dynamic, hidden_dim, kernel_size=(T, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU()
        )
        self.dynamic_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # 3. State Encoder (3D Conv for Fire History)
        # Captures speed/direction of spread over T steps
        self.state_encoder = nn.Sequential(
            nn.Conv3d(n_fire, hidden_dim, kernel_size=(T, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU()
        )
        self.state_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # --- Fusion ---
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # --- Backbone: Swin Transformer (Same as Level 2) ---
        swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        
        original_conv = swin.features[0][0]
        new_conv = nn.Conv2d(
            hidden_dim, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride
        )
        swin.features[0][0] = new_conv
        
        self.swin_features = swin.features
        self.swin_norm = swin.norm

        # --- Decoder (Same as Level 2) ---
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(768, 384)
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(384, 192)
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(192, 96)
        self.final_up = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, doys: Any = None) -> torch.Tensor:
        if x.ndim != 5:
             # Handle edge case where T might be squeezed
            if x.ndim == 4: x = x.unsqueeze(1)
            else: raise ValueError(f"Expected 5D input, got {x.shape}")

        x_permuted = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        
        # 1. Split
        x_static = x_permuted[:, self.static_indices, 0, :, :] # T=0
        x_dynamic = x_permuted[:, self.dynamic_indices, :, :, :] # Full T
        x_state = x_permuted[:, self.fire_index, :, :, :] # Full T

        # 2. Encode
        e_static = self.static_encoder(x_static)
        
        # Dynamic: (B, C, T, H, W) -> Conv3d -> (B, D, 1, H, W) -> Squeeze -> Proj
        e_dynamic = self.dynamic_encoder(x_dynamic).squeeze(2)
        e_dynamic = self.dynamic_proj(e_dynamic)

        # State: (B, 1, T, H, W) -> Conv3d -> (B, D, 1, H, W) -> Squeeze -> Proj
        e_state = self.state_encoder(x_state).squeeze(2)
        e_state = self.state_proj(e_state)

        # 3. Fuse
        e_fused = e_state + (self.alpha * e_static) + (self.beta * e_dynamic)

        # 4. Backbone (Swin)
        x0 = self.swin_features[0](e_fused) 
        x1 = self.swin_features[1](x0) 
        x1_down = self.swin_features[2](x1) 
        x2 = self.swin_features[3](x1_down)
        x2_down = self.swin_features[4](x2) 
        x3 = self.swin_features[5](x2_down)
        x3_down = self.swin_features[6](x3) 
        x4 = self.swin_features[7](x3_down)
        x4 = self.swin_norm(x4) 

        def to_nchw(t): return t.permute(0, 3, 1, 2)
        x1, x2, x3, x4 = to_nchw(x1), to_nchw(x2), to_nchw(x3), to_nchw(x4)

        # 5. Decoder
        d4 = self.up1(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.conv1(d4)
        d3 = self.up2(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv2(d3)
        d2 = self.up3(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.conv3(d2)
        out = self.final_up(d2)
        logits = self.final_conv(out)

        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)