import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, Tuple, Any
from pathlib import Path
import os

def collate_fn(batch):
    return (
        torch.stack([item[0] for item in batch]),  # Pixel values
        [item[1] for item in batch],               # Synapse info
        [item[2] for item in batch]                # Bbox names
    )

def extract_features(model: nn.Module,
                    dataset: Any,
                    batch_size: int = 32,
                    num_workers: int = 8) -> pd.DataFrame:
    """
    Extract features from the dataset using the provided model.
    
    Args:
        model: Neural network model
        dataset: Dataset to extract features from
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataFrame containing extracted features and metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    features = []
    metadata = []

    with torch.no_grad():
        for batch in dataloader:
            pixels, info, names = batch
            inputs = pixels.permute(0, 2, 1, 3, 4).to(device)

            batch_features = model.features(inputs)
            pooled_features = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)
            pooled_features = pooled_features.view(pooled_features.size(0), -1).cpu().numpy()
            
            features.append(pooled_features)
            metadata.extend(zip(names, info))

    # Combine features
    features = np.concatenate(features, axis=0)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in metadata
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
                            drive_dir: str = None) -> str:
    """
    Extract features and save them to CSV files.
    
    Args:
        model: Neural network model
        dataset: Dataset to extract features from
        seg_type: Segmentation type
        alpha: Alpha blending factor
        output_dir: Directory to save the CSV file
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        drive_dir: Optional Google Drive directory to copy the CSV file to
        
    Returns:
        Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract features
    features_df = extract_features(model, dataset, batch_size=batch_size, num_workers=num_workers)

    # Prepare filename and save path
    csv_filename = f"features_seg{seg_type}_alpha{str(alpha).replace('.', '_')}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    # Save features to CSV
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