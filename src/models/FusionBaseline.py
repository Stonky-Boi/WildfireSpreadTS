import torch
import torch.nn as nn
from typing import Any, List, Optional
from .BaseModel import BaseModel

class FusionBaseline(BaseModel):
    """
    Level 1: The Fusion Engine.
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str = "Dice",
        use_doy: bool = False,
        # Matching BaseModel's List type
        required_img_size: Optional[List[int]] = None, 
        hidden_dim: int = 64,
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
            required_img_size=required_img_size, # Pass list directly
            *args,
            **kwargs
        )
        self.save_hyperparameters()

        # Static: Topography (12-14) + Landcover (16-32)
        self.static_indices = [12, 13, 14] + list(range(16, 33))
        # Dynamic: Weather/VIIRS (0-11) + Drought (15) + Forecasts (33-38)
        self.dynamic_indices = list(range(12)) + [15] + list(range(33, 39))
        # State: Active Fire (39)
        self.fire_index = [39]

        num_static = len(self.static_indices)
        num_dynamic = len(self.dynamic_indices)
        num_state = len(self.fire_index)

        self.static_encoder = nn.Sequential(
            nn.Conv2d(num_static, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(num_dynamic, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.state_encoder = nn.Sequential(
            nn.Conv2d(num_state, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.backbone = nn.Sequential(
            SimpleResBlock(hidden_dim),
            SimpleResBlock(hidden_dim),
            SimpleResBlock(hidden_dim)
        )

        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, doys: Any = None) -> torch.Tensor:
        if x.ndim != 5:
            if x.ndim == 4:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"Expected 5D input (B, T, C, H, W), got shape {x.shape}")

        x_permuted = x.permute(0, 2, 1, 3, 4)
        x_static = x_permuted[:, self.static_indices, 0, :, :] 
        x_dynamic = x_permuted[:, self.dynamic_indices, :, :, :].mean(dim=2)
        x_state = x_permuted[:, self.fire_index, :, :, :].mean(dim=2)

        embedding_static = self.static_encoder(x_static)
        embedding_dynamic = self.dynamic_encoder(x_dynamic)
        embedding_state = self.state_encoder(x_state)

        embedding_fused = embedding_state + (self.alpha * embedding_static) + (self.beta * embedding_dynamic)
        features = self.backbone(embedding_fused)
        logits = self.head(features)

        return logits

class SimpleResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out