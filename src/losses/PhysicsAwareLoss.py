import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PhysicsAwareLoss(nn.Module):
    """
    Implements the Physics-Guided Anisotropic Loss Function.
    Combines a boundary-and-wind-weighted Dice Loss with Focal Loss.
    """
    def __init__(
        self, 
        beta: float = 1.0, 
        lambda_focal: float = 0.3, 
        lambda_dice: float = 0.7,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.beta = beta
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.epsilon = epsilon

    def _extract_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Extracts the active fire boundary using morphological dilation and erosion.
        """
        # Ensure targets are float for pooling
        targets_float = targets.float()
        
        # Dilation expands the fire mask outward
        dilation = F.max_pool2d(targets_float, kernel_size=3, stride=1, padding=1)
        
        # Erosion shrinks the fire mask inward (implemented via negative max pooling)
        erosion = -F.max_pool2d(-targets_float, kernel_size=3, stride=1, padding=1)
        
        # The boundary is the difference between the dilated and eroded masks
        boundary_mask = dilation - erosion
        
        return boundary_mask

    def _calculate_spatial_gradients(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the x and y spatial gradients of the fire mask using Sobel filters.
        """
        # Define Sobel kernels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]], device=targets.device).view(1, 1, 3, 3)
                                
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], 
                                [0.0, 0.0, 0.0], 
                                [1.0, 2.0, 1.0]], device=targets.device).view(1, 1, 3, 3)

        # Apply convolutions to find gradients
        grad_x = F.conv2d(targets.float(), sobel_x, padding=1)
        grad_y = F.conv2d(targets.float(), sobel_y, padding=1)

        return grad_x, grad_y

    def _compute_wind_advection(self, targets: torch.Tensor, wind_u: torch.Tensor, wind_v: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized advection field based on wind alignment.
        """
        grad_x, grad_y = self._calculate_spatial_gradients(targets)
        
        # Dot product of spatial gradient and wind vector
        advection = (grad_x * wind_u) + (grad_y * wind_v)
        
        # Normalize to [0, 1] per item in the batch
        batch_size = advection.size(0)
        advection_flat = advection.view(batch_size, -1)
        
        min_val = advection_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = advection_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        
        normalized_advection = (advection - min_val) / (max_val - min_val + self.epsilon)
        
        return normalized_advection

    def _compute_focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
        """
        Computes the standard Focal Loss for binary classification.
        """
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        probabilities = torch.sigmoid(predictions)
        
        # Compute focal weight: (1 - p)^gamma for positive class, p^gamma for negative class
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma
        
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        wind_u: Optional[torch.Tensor] = None, 
        wind_v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the total physics-aware loss.
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} and targets shape {targets.shape} must match.")
            
        if wind_u is None or wind_v is None:
            raise ValueError("wind_u and wind_v tensors must be provided to compute the anisotropic loss.")

        # Ensure targets have a channel dimension for spatial operations
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if predictions.ndim == 3:
            predictions = predictions.unsqueeze(1)
            
        # 1. Calculate Boundary Weight Map
        boundary_mask = self._extract_boundary_mask(targets)
        
        # 2. Calculate Wind Alignment Map
        advection_map = self._compute_wind_advection(targets, wind_u, wind_v)
        
        # 3. Combine into Final Anisotropic Weight Map
        # FIX: We MUST add 1.0 to the entire map so background false positives are penalized.
        # The boundary mask applies an EXTRA penalty (up to 2.0x) where the wind is blowing.
        weight_map = 1.0 + (boundary_mask * (1.0 + self.beta * advection_map))
        
        # 4. Compute Weighted Dice Loss
        probabilities = torch.sigmoid(predictions)
        
        intersection = torch.sum(weight_map * probabilities * targets)
        denominator = torch.sum(weight_map * probabilities) + torch.sum(weight_map * targets) + self.epsilon
        
        dice_loss = 1.0 - (2.0 * intersection / denominator)
        
        # 5. Compute Focal Loss
        focal_loss = self._compute_focal_loss(predictions, targets)
        
        # 6. Final Multi-Objective Combination
        total_loss = (self.lambda_focal * focal_loss) + (self.lambda_dice * dice_loss)
        
        return total_loss