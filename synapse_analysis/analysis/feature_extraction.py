import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Union
from pathlib import Path
import os
import warnings

def apply_global_normalization(tensor_batch: torch.Tensor, global_stats: Dict[str, List[float]]) -> torch.Tensor:
    """
    Apply global normalization to a batch of tensors.
    
    This function is deprecated and will be removed in a future version.
    It now falls back to standard normalization with default values.
    
    Args:
        tensor_batch: Batch of tensors to normalize [B, C, D, H, W]
        global_stats: Dictionary containing 'mean' and 'std' values
        
    Returns:
        Normalized tensor batch
    """
    warnings.warn(
        "apply_global_normalization is deprecated and will be removed in a future version. "
        "Using standard normalization instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    
    if not isinstance(tensor_batch, torch.Tensor):
        tensor_batch = torch.tensor(tensor_batch)
        
    # Use default normalization values
    mean = torch.tensor([0.485]).view(1, -1, 1, 1, 1).to(tensor_batch.device)
    std = torch.tensor([0.229]).view(1, -1, 1, 1, 1).to(tensor_batch.device)
    
    # Apply normalization
    normalized_batch = (tensor_batch - mean) / std
    
    return normalized_batch

def collate_fn(batch):
    return (
        torch.stack([item[0] for item in batch]),  # Pixel values
        [item[1] for item in batch],               # Synapse info
        [item[2] for item in batch]                # Bbox names
    )

def extract_features(model: nn.Module,
                    dataset: Any,
                    batch_size: int = 32,
                    num_workers: int = 8,
                    apply_global_norm: bool = False,  # Kept for backward compatibility but not used
                    global_stats: Dict[str, List[float]] = None) -> pd.DataFrame:  # Kept for backward compatibility but not used
    """
    Extract features from a dataset using a pre-trained model.
    
    Args:
        model: Pre-trained model for feature extraction
        dataset: Dataset to extract features from
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        apply_global_norm: Whether to apply global normalization before feature extraction (deprecated, not used)
        global_stats: Dictionary containing global 'mean' and 'std' values (deprecated, not used)
        
    Returns:
        DataFrame containing extracted features
    """
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Extract features
    features = []
    synapse_info = []
    bbox_names = []
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs, info, bbox = batch
            
            # Move inputs to the same device as the model
            inputs = inputs.to(device)
            
            # Reshape inputs to match model expectations
            # Model expects [batch_size, channels, depth, height, width]
            # But inputs are [batch_size, depth, channels, height, width]
            # So we need to permute the dimensions
            if inputs.dim() == 4:  # [batch_size, depth, height, width]
                # Add channel dimension
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() == 5:  # [batch_size, depth, channels, height, width]
                # Permute to [batch_size, channels, depth, height, width]
                inputs = inputs.permute(0, 2, 1, 3, 4)
            
            # Extract features using the feature extractor part of the model
            # This avoids the classifier which might have dimension mismatch
            features_tensor = model.features(inputs)
            
            # Apply global average pooling to get a fixed-size feature vector
            pooled_features = torch.nn.functional.adaptive_avg_pool3d(features_tensor, (1, 1, 1))
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            
            # Add to lists
            features.append(pooled_features.cpu().numpy())
            synapse_info.extend(info)
            bbox_names.extend(bbox)

    # Combine features
    features = np.concatenate(features, axis=0)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in zip(bbox_names, synapse_info)
    ])

    # Create feature columns
    feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)

    # Combine metadata and features
    combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    return combined_df

def extract_and_save_features(model: nn.Module,
                            dataset: Any,
                            seg_type: int,
                            alpha: float,
                            output_dir: str,
                            batch_size: int = 4,
                            num_workers: int = 2,
                            apply_global_norm: bool = False,  # Kept for backward compatibility but not used
                            global_stats: Dict[str, List[float]] = None,  # Kept for backward compatibility but not used
                            drive_dir: str = None) -> str:
    """
    Extract features from a dataset and save them to a CSV file.
    
    Args:
        model: Pre-trained model for feature extraction
        dataset: Dataset to extract features from
        seg_type: Segmentation type used
        alpha: Alpha value used
        output_dir: Directory to save the CSV file
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        apply_global_norm: Whether to apply global normalization before feature extraction (deprecated, not used)
        global_stats: Dictionary containing global 'mean' and 'std' values (deprecated, not used)
        drive_dir: Google Drive directory to save a copy of the CSV file (optional)
        
    Returns:
        Path to the saved CSV file
    """
    # Extract features
    features_df = extract_features(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_global_norm=False,  # Always use False
        global_stats=None  # Always use None
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features to CSV
    norm_suffix = ""  # No longer using global normalization suffix
    csv_filename = f"features_segtype_{seg_type}_alpha_{alpha:.1f}{norm_suffix}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    features_df.to_csv(csv_filepath, index=False)
    
    print(f"Features for SegType {seg_type} and Alpha {alpha} saved to {csv_filepath}")

    # Copy to Google Drive if specified
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        drive_path = os.path.join(drive_dir, csv_filename)
        features_df.to_csv(drive_path, index=False)
        print(f"CSV file copied to: {drive_path}")

    return csv_filepath

def load_features(csv_filepath: str) -> pd.DataFrame:
    """
    Load features from a CSV file.
    
    Args:
        csv_filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded features
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Feature file not found: {csv_filepath}")
        
    return pd.read_csv(csv_filepath) 