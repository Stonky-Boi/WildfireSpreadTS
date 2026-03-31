import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any, List, Optional
from .BaseModel import BaseModel

try:
    from mamba_ssm import Mamba2
    import causal_conv1d
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class PhysicsFusionMambaModel(BaseModel):
    """
    Level 6: Physics-Aware Hierarchical Decomposition.
    
    Integrates Spatiotemporal Mamba with structured, physics-guided feature fusion.
    Terrain acts as a spatial constraint mask, weather acts as a directional driver,
    and both modulate the anchor fire state.
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
        n_leading_observations: int = 5,
        patch_size: int = 4,
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

        # --- Features ---
        self.static_indices = [12, 13, 14] + list(range(16, 33))
        self.dynamic_indices = list(range(12)) + [15] + list(range(33, 39))
        self.fire_index = [39]

        n_static = len(self.static_indices)
        n_dynamic = len(self.dynamic_indices)
        n_fire = len(self.fire_index)

        # 1. Modality-Specific Encoders 
        self.static_encoder = nn.Sequential(
            nn.Conv2d(n_static, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.dynamic_proj_in = nn.Conv3d(n_dynamic, hidden_dim, kernel_size=1)
        self.dynamic_mamba = SpatiotemporalMambaBlock(hidden_dim, dim=hidden_dim, patch_size=patch_size)
        self.dynamic_proj_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self.state_proj_in = nn.Conv3d(n_fire, hidden_dim, kernel_size=1)
        self.state_mamba = SpatiotemporalMambaBlock(hidden_dim, dim=hidden_dim, patch_size=patch_size)
        self.state_proj_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # --- Level 6: Physics-Guided Composition ---
        
        # Terrain Conditioning Function T(E_terrain) -> Spatial Mask 
        self.terrain_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid() # Creates a mask in [0, 1] 
        )
        
        # Weather Modulation Function W(E_weather) -> Spatial Mask 
        self.weather_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()
        )

        # Learnable scalars balancing terrain constraints vs weather drivers 
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # --- Multi-Scale Spatial Reasoning (Swin) ---
        swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        swin.features[0][0] = nn.Conv2d(hidden_dim, 96, kernel_size=4, stride=4)
        self.swin_features = swin.features
        self.swin_norm = swin.norm

        # --- Decoder ---
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
            if x.ndim == 4: x = x.unsqueeze(1)
            else: raise ValueError(f"Expected 5D input, got {x.shape}")

        x_permuted = x.permute(0, 2, 1, 3, 4) 
        
        # Semantic Decomposition 
        x_static = x_permuted[:, self.static_indices, 0, :, :] 
        x_dynamic = x_permuted[:, self.dynamic_indices, :, :, :]
        x_state = x_permuted[:, self.fire_index, :, :, :]

        # Encode (Stage 1) 
        e_terrain = self.static_encoder(x_static)

        dyn_feat = self.dynamic_proj_in(x_dynamic)
        dyn_feat = self.dynamic_mamba(dyn_feat) 
        e_weather = self.dynamic_proj_out(dyn_feat[:, :, -1, :, :])

        state_feat = self.state_proj_in(x_state)
        state_feat = self.state_mamba(state_feat)
        e_state = self.state_proj_out(state_feat[:, :, -1, :, :])

        # --- Stage 2: Physics-Guided Fusion  ---
        
        # 1. Terrain Constraint: sigma(W_T * E_terrain) * E_fire 
        terrain_mask = self.terrain_gate(e_terrain)
        constrained_fire = terrain_mask * e_state
        
        # 2. Weather Driver: sigma(W_W * E_weather) * E_fire 
        weather_mask = self.weather_gate(e_weather)
        driven_fire = weather_mask * e_state
        
        # 3. Additive Fusion: Anchored on the current fire state 
        e_fused = e_state + (self.alpha * constrained_fire) + (self.beta * driven_fire) 
        
        # Swin Backbone
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

        # Decode
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

class SpatiotemporalMambaBlock(nn.Module):
    def __init__(self, in_channels, dim, patch_size=4, d_state=16, d_conv=4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv3d(
            in_channels, dim, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )
        self.has_mamba = HAS_MAMBA
        if self.has_mamba:
            self.mamba = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)
        else:
            self.mamba = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)
            self.norm = nn.LayerNorm(dim)
        self.patch_unembed = nn.ConvTranspose3d(
            dim, in_channels, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_patched = self.patch_embed(x)
        _, D, _, Hp, Wp = x_patched.shape
        x_seq = x_patched.permute(0, 2, 3, 4, 1).contiguous().view(B, T * Hp * Wp, D)
        if self.has_mamba:
            out_seq = self.mamba(x_seq)
        else:
            out_seq, _ = self.mamba(x_seq)
            out_seq = self.norm(out_seq)
        out_patched = out_seq.view(B, T, Hp, Wp, D).permute(0, 4, 1, 2, 3).contiguous()
        out = self.patch_unembed(out_patched)
        return out + x

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