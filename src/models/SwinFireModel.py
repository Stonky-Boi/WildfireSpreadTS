import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any, List, Optional
from .BaseModel import BaseModel

class SwinFireModel(BaseModel):
    """
    Level 2: Swin Transformer Backbone.
    
    Upgrades the Level 1 Fusion Engine by replacing the simple ResNet backbone 
    with a Swin Transformer (Tiny) and a U-Net style decoder.
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
        *args: Any,
        **kwargs: Any
    ):
        if required_img_size is None:
            required_img_size = [128, 128]

        # Enforce flatten_temporal_dimension=False to handle T dimension manually in encoders
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

        # --- Encoders (Same as Level 1) ---
        self.static_encoder = nn.Sequential(
            nn.Conv2d(n_static, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # Dynamic Encoder: Still simple averaging for now (Level 3 will upgrade this)
        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(n_dynamic, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.dynamic_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # State Encoder
        self.state_encoder = nn.Sequential(
            nn.Conv2d(n_fire, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.state_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # --- Fusion Parameters ---
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # --- Level 2 Upgrade: Swin Transformer Backbone ---
        # We use swin_t (Tiny). 
        swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        
        # 1. Modify the first layer (Patch Partition) to accept 'hidden_dim' channels
        # torchvision Swin features[0] is a Sequential(Conv2d, Permute, LayerNorm)
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

        # --- Decoder (U-Net style) ---
        # Swin T channel sizes: 96 -> 192 -> 384 -> 768
        
        # Stage 4 (768) -> Stage 3 (384)
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(768, 384) # 384 + 384

        # Stage 3 -> Stage 2 (192)
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(384, 192)

        # Stage 2 -> Stage 1 (96)
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(192, 96)

        # Final prediction: 96 -> H,W
        self.final_up = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, doys: Any = None) -> torch.Tensor:
        if x.ndim != 5:
            if x.ndim == 4: x = x.unsqueeze(1)
            else: raise ValueError(f"Expected 5D input, got {x.shape}")

        x_permuted = x.permute(0, 2, 1, 3, 4)
        
        # 1. Feature Splitting
        x_static = x_permuted[:, self.static_indices, 0, :, :]
        x_dynamic = x_permuted[:, self.dynamic_indices, :, :, :].mean(dim=2)
        x_state = x_permuted[:, self.fire_index, :, :, :].mean(dim=2)

        # 2. Encoders
        e_static = self.static_encoder(x_static)
        e_dynamic = self.dynamic_proj(self.dynamic_encoder(x_dynamic))
        e_state = self.state_proj(self.state_encoder(x_state))

        # 3. Fusion
        e_fused = e_state + (self.alpha * e_static) + (self.beta * e_dynamic)

        # 4. Swin Backbone
        # Swin in torchvision expects inputs, but internally moves to NHWC. 
        # features[0] (PatchEmbed) handles the permute.
        
        # Stage 0: Patch Partition
        x0 = self.swin_features[0](e_fused) # Out: (B, H/4, W/4, 96) [NHWC]
        
        # Stage 1
        x1 = self.swin_features[1](x0) 
        x1_down = self.swin_features[2](x1) # Patch Merging -> (B, H/8, W/8, 192)

        # Stage 2
        x2 = self.swin_features[3](x1_down)
        x2_down = self.swin_features[4](x2) # -> (B, H/16, W/16, 384)

        # Stage 3
        x3 = self.swin_features[5](x2_down)
        x3_down = self.swin_features[6](x3) # -> (B, H/32, W/32, 768)

        # Stage 4
        x4 = self.swin_features[7](x3_down)
        x4 = self.swin_norm(x4) # (B, H/32, W/32, 768)

        # Helper to convert NHWC back to NCHW for Decoder
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