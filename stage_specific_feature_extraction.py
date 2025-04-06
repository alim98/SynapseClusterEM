#!/usr/bin/env python3
"""
Stage-Specific Feature Extraction Script

This script demonstrates how to use the VGG3DStageExtractor class to extract features
from specific stages or layers of the VGG3D model, with a focus on layer 20.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import umap

# Import from newdl module instead of synapse
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from synapse import (
    Vgg3D, 
    load_model_from_checkpoint,
    config
)
from vgg3d_stage_extractor import VGG3DStageExtractor
from inference import load_and_prepare_data

def extract_layer_20_features(model, dataset, output_dir, batch_size=2):
    """
    Extract features specifically from layer 20 of the VGG3D model.
    
    Args:
        model: The VGG3D model
        dataset: The dataset to extract features from
        output_dir: Directory to save the extracted features
        batch_size: Batch size for the dataloader
        
    Returns:
        DataFrame with extracted features and metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Create the stage extractor
    extractor = VGG3DStageExtractor(model)
    
    # Get information about the stages for logging
    stage_info = extractor.get_stage_info()
    print("VGG3D Stage Information:")
    for stage_num, info in stage_info.items():
        print(f"Stage {stage_num}: Layers {info['range'][0]}-{info['range'][1]}")
        print("Containing layers:")
        for idx, layer_info in info['layers']:
            print(f"  Layer {idx}: {layer_info}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )
    
    # Extract features from layer 20
    features = []
    metadata = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Skip empty batches (can happen if all items in a batch were discarded)
            if len(batch[0]) == 0:
                continue
                
            pixels, info, names = batch
            inputs = pixels.permute(0, 2, 1, 3, 4).to(device)
            
            # Extract features using the extractor's layer 20 method
            batch_features = extractor.extract_layer_20(inputs)
            
            # Global average pooling to get a feature vector
            batch_size = batch_features.shape[0]
            num_channels = batch_features.shape[1]
            spatial_size = np.prod(batch_features.shape[2:])
            
            # Reshape to (batch_size, channels, -1) for easier processing
            batch_features_reshaped = batch_features.reshape(batch_size, num_channels, -1)
            
            # Global average pooling across spatial dimensions
            pooled_features = torch.mean(batch_features_reshaped, dim=2)
            
            # Convert to numpy
            features_np = pooled_features.cpu().numpy()
            
            features.append(features_np)
            metadata.extend(zip(names, info))
    
    # Concatenate all features
    features = np.concatenate(features, axis=0)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in metadata
    ])
    
    # Create feature DataFrame
    feature_columns = [f'layer20_feat_{i+1}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)
    
    # Combine metadata and features
    combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = "layer20_features.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    combined_df.to_csv(csv_filepath, index=False)
    print(f"Layer 20 features saved to {csv_filepath}")
    
    # Perform UMAP for visualization
    feature_cols = [c for c in combined_df.columns if c.startswith('layer20_feat_')]
    features_for_umap = combined_df[feature_cols].values
    
    # Scale features
    features_scaled = StandardScaler().fit_transform(features_for_umap)
    
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)
    
    # Add UMAP coordinates to DataFrame
    combined_df['umap_x'] = umap_results[:, 0]
    combined_df['umap_y'] = umap_results[:, 1]
    
    # Save updated DataFrame
    combined_df.to_csv(csv_filepath, index=False)
    print(f"Updated with UMAP coordinates and saved to {csv_filepath}")
    
    return combined_df

def extract_all_stages_features(model, dataset, output_dir, batch_size=2):
    """
    Extract features from all stages of the VGG3D model for comparison.
    
    Args:
        model: The VGG3D model
        dataset: The dataset to extract features from
        output_dir: Directory to save the extracted features
        batch_size: Batch size for the dataloader
        
    Returns:
        Dictionary mapping stage numbers to DataFrames with extracted features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Create the stage extractor
    extractor = VGG3DStageExtractor(model)
    
    # Get information about the stages
    stage_info = extractor.get_stage_info()
    stage_nums = list(stage_info.keys())
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )
    
    # Dictionary to store results
    results = {}
    
    # Process each batch
    metadata_list = []
    
    for stage_num in stage_nums:
        print(f"Extracting features for Stage {stage_num}...")
        features = []
        metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Skip empty batches (can happen if all items in a batch were discarded)
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)
                
                # Extract features for this stage
                batch_features = extractor.extract_stage(stage_num, inputs)
                
                # Global average pooling
                batch_size = batch_features.shape[0]
                num_channels = batch_features.shape[1]
                spatial_size = np.prod(batch_features.shape[2:])
                
                # Reshape and pool
                batch_features_reshaped = batch_features.reshape(batch_size, num_channels, -1)
                pooled_features = torch.mean(batch_features_reshaped, dim=2)
                
                # Convert to numpy
                features_np = pooled_features.cpu().numpy()
                
                features.append(features_np)
                
                # Only collect metadata once
                if stage_num == 1 and batch_idx == 0:
                    metadata_list.extend(zip(names, info))
        
        # Concatenate features for this stage
        stage_features = np.concatenate(features, axis=0)
        
        # Create feature DataFrame
        feature_columns = [f'stage{stage_num}_feat_{i+1}' for i in range(stage_features.shape[1])]
        stage_df = pd.DataFrame(stage_features, columns=feature_columns)
        
        results[stage_num] = stage_df
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in metadata_list
    ])
    
    # Combine all stage features with metadata
    combined_df = metadata_df.copy()
    for stage_num, stage_df in results.items():
        combined_df = pd.concat([combined_df, stage_df], axis=1)
    
    # Save combined results
    os.makedirs(output_dir, exist_ok=True)
    csv_filepath = os.path.join(output_dir, "all_stages_features.csv")
    combined_df.to_csv(csv_filepath, index=False)
    print(f"Features from all stages saved to {csv_filepath}")
    
    return results

def load_model_and_extract_stage_specific_features(dataset, features_df, device="cuda", layer_num=20):
    """
    Load the VGG3D model and extract stage-specific features (primarily layer 20)
    
    Args:
        dataset: The dataset to extract features from
        features_df: DataFrame to add features to
        device: Device to run the model on ("cuda" or "cpu")
        layer_num: The specific layer to extract features from (default: 20)
        
    Returns:
        DataFrame with extracted features and metadata
    """
    # Initialize the VGG3D model
    model = Vgg3D(
        input_size=(80, 80, 80),  # Adjust dimensions based on your dataset
        fmaps=24,
        downsample_factors=[(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        fmap_inc=(2, 2, 2, 2),
        n_convolutions=(4, 2, 2, 2),
        output_classes=2,  # Adjust based on your classes
        input_fmaps=1
    )
    
    # Find the checkpoint path
    checkpoint_path = config.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Try to find a checkpoint file
        possible_checkpoint_paths = [
            os.path.join("checkpoints", "vgg3d_model.pth"),
            os.path.join("models", "vgg3d_model.pth"),
            os.path.join("models", "checkpoint.pth"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "vgg3d_model.pth"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "vgg3d_model.pth")
        ]
        
        for cp_path in possible_checkpoint_paths:
            if os.path.exists(cp_path):
                checkpoint_path = cp_path
                break
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found. Please provide a valid checkpoint path.")
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(model, checkpoint_path)
    print(f"Model loaded successfully from {checkpoint_path}")
    
    # Set up the device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(device).eval()
    
    # Create the stage extractor
    extractor = VGG3DStageExtractor(model)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )
    
    # Extract features from the specified layer
    features = []
    bbox_names = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Skip empty batches (can happen if all items in a batch were discarded)
            if len(batch[0]) == 0:
                continue
                
            pixels, info, names = batch
            inputs = pixels.permute(0, 2, 1, 3, 4).to(device)
            
            # Extract features using the specified layer
            if layer_num == 20:
                batch_features = extractor.extract_layer_20(inputs)
            else:
                # For other layers, add code here or raise an error
                raise ValueError(f"Feature extraction for layer {layer_num} is not implemented")
            
            # Global average pooling to get a feature vector
            batch_size = batch_features.shape[0]
            num_channels = batch_features.shape[1]
            
            # Reshape to (batch_size, channels, -1) for easier processing
            batch_features_reshaped = batch_features.reshape(batch_size, num_channels, -1)
            
            # Global average pooling across spatial dimensions
            pooled_features = torch.mean(batch_features_reshaped, dim=2)
            
            # Convert to numpy
            features_np = pooled_features.cpu().numpy()
            
            features.append(features_np)
            bbox_names.extend(names)
    
    # Concatenate all features
    features_array = np.concatenate(features, axis=0) if features else np.array([])
    
    if len(features_array) == 0:
        print("No features extracted")
        return features_df
    
    # Create feature columns
    feature_columns = [f"feature_{i}" for i in range(features_array.shape[1])]
    
    # Create a DataFrame with the extracted features
    feature_df = pd.DataFrame(features_array, columns=feature_columns)
    
    # Add a column for bbox names to use for joining
    feature_df['bbox'] = bbox_names
    
    # Merge with the existing features DataFrame
    if 'bbox' in features_df.columns:
        # Use bbox column for merging if available
        merged_df = pd.merge(features_df, feature_df, on='bbox', how='left')
    else:
        # Otherwise, just append the features (assuming same order)
        assert len(feature_df) == len(features_df), "Feature count mismatch"
        for col in feature_columns:
            features_df[col] = feature_df[col].values
        merged_df = features_df
    
    print(f"Extracted {features_array.shape[1]} features from layer {layer_num}")
    return merged_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stage-Specific Feature Extraction")
    parser.add_argument('--output_dir', type=str, default='results/stage_features',
                      help='Directory to save extracted features')
    parser.add_argument('--extract_all_stages', action='store_true',
                      help='Extract features from all stages, not just layer 20')
    parser.add_argument('--segmentation_type', type=int, default=None,
                      help='Segmentation type to use (defaults to config value)')
    parser.add_argument('--alpha', type=float, default=None,
                      help='Alpha value for feature extraction (defaults to config value)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for dataloader')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Override config values if provided
    if args.segmentation_type is not None:
        config.segmentation_type = args.segmentation_type
    
    if args.alpha is not None:
        config.alpha = args.alpha
    
    # Initialize model
    print("Initializing model...")
    model = Vgg3D()
    
    # Load checkpoint if available
    if hasattr(config, 'model_checkpoint') and os.path.exists(config.model_checkpoint):
        model = load_model_from_checkpoint(model, config.model_checkpoint)
    
    # Load data
    print("Loading data...")
    vol_data_dict, syn_df = load_and_prepare_data(config)
    processor = Synapse3DProcessor(size=config.size)
    
    # Create dataset
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha
    )
    
    print(f"Created dataset with segmentation_type={config.segmentation_type}, alpha={config.alpha}")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Extract features
    if args.extract_all_stages:
        print("Extracting features from all stages...")
        results = extract_all_stages_features(model, dataset, args.output_dir, args.batch_size)
        print(f"Extracted features from {len(results)} stages")
    else:
        print("Extracting features from layer 20...")
        features_df = extract_layer_20_features(model, dataset, args.output_dir, args.batch_size)
        print(f"Extracted {features_df.shape[1] - len(features_df.columns[~features_df.columns.str.contains('layer20_feat_')])} features from layer 20")
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main() 