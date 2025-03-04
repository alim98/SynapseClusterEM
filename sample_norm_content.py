import torch
import numpy as np
from typing import Union

def apply_sample_normalization(tensor_data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Apply sample-wise normalization to tensor data.
    
    This function normalizes each sample (image) independently, using the mean
    and standard deviation calculated from that specific sample.
    
    Args:
        tensor_data: Input tensor data to normalize [B, C, ...] or [C, ...]
        
    Returns:
        Normalized tensor data
    """
    # Convert to tensor if numpy array
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data)
    
    # Handle different tensor dimensions
    if tensor_data.dim() == 5:  # [B, C, D, H, W]
        # Calculate mean and std over spatial dimensions, keeping batch and channel dims
        mean = tensor_data.mean(dim=(2, 3, 4), keepdim=True)
        std = tensor_data.std(dim=(2, 3, 4), keepdim=True)
    elif tensor_data.dim() == 4:  # [B, C, H, W]
        mean = tensor_data.mean(dim=(2, 3), keepdim=True)
        std = tensor_data.std(dim=(2, 3), keepdim=True)
    elif tensor_data.dim() == 3:  # [C, H, W]
        mean = tensor_data.mean(dim=(1, 2), keepdim=True)
        std = tensor_data.std(dim=(1, 2), keepdim=True)
    else:
        raise ValueError(f"Unsupported tensor dimensions: {tensor_data.dim()}")
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-5)
    
    # Apply normalization
    normalized_data = (tensor_data - mean) / std
    
    return normalized_data 