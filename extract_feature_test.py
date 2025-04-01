"""
Feature extraction test script to replicate previously extracted features.

This script extracts features from a specific synapse sample (bbox1 non_spine_synapsed_056)
and compares its layer20_feat_1 value with the reference in check.xlsx.
"""

import os
import pandas as pd
import numpy as np
import torch
import glob
import warnings
import traceback

from torch.utils.data import DataLoader
import torch.nn as nn

# Import the necessary modules
from synapse import config
from inference import VGG3D
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from vgg3d_stage_extractor import VGG3DStageExtractor

def load_check_file(file_path="check.xlsx"):
    """Load the reference features from Excel file."""
    print(f"Loading reference features from {file_path}")
    try:
        # Suppress openpyxl warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(file_path, engine='openpyxl')
        
        print(f"Successfully loaded reference features with shape: {df.shape}")
        # Print the specific sample and feature we're interested in
        if not df.empty:
            target_sample = df[df['Var1'] == 'non_spine_synapsed_056']
            if not target_sample.empty:
                print("Target sample found in reference data:")
                print(f"Var1: {target_sample['Var1'].values[0]}")
                print(f"bbox: {target_sample['bbox'].values[0]}")
                if 'layer20_feat_1' in target_sample:
                    print(f"layer20_feat_1 value: {target_sample['layer20_feat_1'].values[0]}")
                else:
                    print("Warning: layer20_feat_1 column not found in reference data")
            else:
                print("Warning: Target sample 'non_spine_synapsed_056' not found in reference data")
        return df
    except Exception as e:
        print(f"Error loading reference file: {e}")
        traceback.print_exc()
        return None

def load_specific_sample(sample_name="non_spine_synapsed_056", bbox_name="bbox1"):
    """
    Load a specific sample for feature extraction testing.
    
    Args:
        sample_name: Name of the sample (Var1 column) (default: non_spine_synapsed_056)
        bbox_name: Name of the bbox (default: bbox1)
        
    Returns:
        Tuple of (vol_data_dict, sample_df)
    """
    print(f"Loading specific sample: {bbox_name} {sample_name}")
    
    # Initialize dataloader
    dataloader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load the volume for the specific bbox
    raw_vol, seg_vol, add_mask_vol = dataloader.load_volumes(bbox_name)
    if raw_vol is None:
        print(f"Error: Could not load volume for {bbox_name}")
        return None, None
    
    vol_data_dict = {bbox_name: (raw_vol, seg_vol, add_mask_vol)}
    print(f"Loaded volume for {bbox_name} with shapes: raw={raw_vol.shape}, seg={seg_vol.shape}, add_mask={add_mask_vol.shape}")
    
    # Load synapse metadata from Excel file
    excel_path = os.path.join(config.excel_file, f"{bbox_name}.xlsx")
    if os.path.exists(excel_path):
        print(f"Loading synapse metadata from {excel_path}")
        syn_df = pd.read_excel(excel_path)
        syn_df['bbox_name'] = bbox_name  # Add bbox_name column if not present
        print(f"Loaded synapse metadata with {len(syn_df)} rows")
    else:
        print(f"Warning: Excel file {excel_path} not found")
        # Create a simple DataFrame with minimal information
        syn_df = pd.DataFrame({
            'bbox_name': [bbox_name],
            'Var1': [sample_name]
        })
    
    # Filter to only include the specific sample
    if 'Var1' in syn_df.columns:
        sample_df = syn_df[syn_df['Var1'] == sample_name]
    else:
        print("Warning: 'Var1' column not found in synapse metadata")
        # Try to find a different identifier column
        sample_df = syn_df.iloc[[0]]
    
    if len(sample_df) == 0:
        print(f"Error: Could not find sample {sample_name} in {bbox_name}")
        return vol_data_dict, syn_df.iloc[[0]]
    
    # Verify that we have coordinate information
    required_coords = ['central_coord_1', 'central_coord_2', 'central_coord_3',
                       'side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3',
                       'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3']
    
    missing_coords = [col for col in required_coords if col not in sample_df.columns]
    if missing_coords:
        print(f"Warning: Missing coordinate columns: {missing_coords}")
        print("Using default coordinates at center of volume")
        
        # Add default coordinates at the center of the volume
        center_x = raw_vol.shape[0] // 2
        center_y = raw_vol.shape[1] // 2
        center_z = raw_vol.shape[2] // 2
        
        # Generate coordinates around the center
        for col in missing_coords:
            if col == 'central_coord_1':
                sample_df['central_coord_1'] = center_x
            elif col == 'central_coord_2':
                sample_df['central_coord_2'] = center_y
            elif col == 'central_coord_3':
                sample_df['central_coord_3'] = center_z
            elif col == 'side_1_coord_1':
                sample_df['side_1_coord_1'] = center_x - 5
            elif col == 'side_1_coord_2':
                sample_df['side_1_coord_2'] = center_y
            elif col == 'side_1_coord_3':
                sample_df['side_1_coord_3'] = center_z
            elif col == 'side_2_coord_1':
                sample_df['side_2_coord_1'] = center_x + 5
            elif col == 'side_2_coord_2':
                sample_df['side_2_coord_2'] = center_y
            elif col == 'side_2_coord_3':
                sample_df['side_2_coord_3'] = center_z
    
    print(f"Found specific sample. Metadata: {sample_df.iloc[0].to_dict()}")
    return vol_data_dict, sample_df

def extract_features_with_config(
    vol_data_dict, sample_df, 
    seg_type=10, alpha=1, extraction_method="stage_specific", layer_num=20,
    normalize_volume=False, normalize_across_volume=False,
    smart_crop=False, presynapse_weight=0.5,
    normalize_presynapse_size=False, target_percentage=None,
    size_tolerance=0.1
):
    """
    Extract features from a specific sample with the given configuration.
    
    Args:
        vol_data_dict: Dictionary of volumes
        sample_df: DataFrame with the specific sample
        seg_type: Segmentation type (default: 10)
        alpha: Alpha value (default: 1)
        extraction_method: Feature extraction method (default: 'stage_specific')
        layer_num: Layer number (default: 20)
        normalize_volume: Whether to normalize volumes (default: False)
        normalize_across_volume: Whether to normalize across volume (default: False)
        smart_crop: Whether to enable intelligent cropping (default: False)
        presynapse_weight: Weight for presynapse (default: 0.5)
        normalize_presynapse_size: Enable presynapse size normalization (default: False)
        target_percentage: Target percentage (default: None)
        size_tolerance: Size tolerance (default: 0.1)
        
    Returns:
        DataFrame with extracted features
    """
    print(f"Extracting features with configuration:")
    print(f"  Segmentation Type: {seg_type}")
    print(f"  Alpha: {alpha}")
    print(f"  Extraction Method: {extraction_method}")
    print(f"  Layer Number: {layer_num}")
    print(f"  Normalize Volume: {normalize_volume}")
    print(f"  Normalize Across Volume: {normalize_across_volume}")
    print(f"  Smart Crop: {smart_crop}")
    print(f"  Presynapse Weight: {presynapse_weight}")
    print(f"  Normalize Presynapse Size: {normalize_presynapse_size}")
    print(f"  Target Percentage: {target_percentage}")
    print(f"  Size Tolerance: {size_tolerance}")
    
    # Initialize processor with specified normalization
    processor = Synapse3DProcessor(size=config.size)
    processor.normalize_volume = normalize_volume
    
    # Create dataset with the specific sample
    dataset = SynapseDataset(
        vol_data_dict, 
        sample_df,
        processor=processor,
        segmentation_type=seg_type,
        alpha=alpha,
        normalize_across_volume=normalize_across_volume,
        smart_crop=smart_crop,
        presynapse_weight=presynapse_weight,
        normalize_presynapse_size=normalize_presynapse_size,
        target_percentage=target_percentage,
        size_tolerance=size_tolerance
    )
    
    # Load model
    model = VGG3D()
    
    # Extract features
    if extraction_method == 'stage_specific':
        # Create a VGG3D stage extractor
        extractor = VGG3DStageExtractor(model)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=lambda b: (
                torch.stack([item[0] for item in b]),
                [item[1] for item in b],
                [item[2] for item in b]
            )
        )
        
        # Extract features from specific layer
        features = []
        metadata = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        with torch.no_grad():
            for batch in dataloader:
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)
                
                # Extract features from specified layer
                batch_features = extractor.extract_layer(layer_num, inputs)
                
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
                metadata.extend(zip(names, info))
        
        # Concatenate all features
        features = np.concatenate(features, axis=0)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])
        
        # Create feature DataFrame
        feature_columns = [f'layer{layer_num}_feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        # Combine metadata and features
        combined_df = pd.concat([metadata_df, features_df], axis=1)
    else:
        # Use standard extraction for completeness (though we know this isn't what we want)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=lambda b: (
                torch.stack([item[0] for item in b]),
                [item[1] for item in b],
                [item[2] for item in b]
            )
        )

        features = []
        metadata = []

        with torch.no_grad():
            for batch in dataloader:
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)

                batch_features = model.features(inputs)
                pooled_features = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)

                batch_features_np = pooled_features.cpu().numpy()
                batch_size = batch_features_np.shape[0]
                num_features = np.prod(batch_features_np.shape[1:])
                batch_features_np = batch_features_np.reshape(batch_size, num_features)
                
                features.append(batch_features_np)
                metadata.extend(zip(names, info))

        features = np.concatenate(features, axis=0)

        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])

        feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)

        combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    print(f"Extracted features with shape: {combined_df.shape}")
    return combined_df

def compare_feature_value(extracted_df, reference_df, feature_column='layer20_feat_1'):
    """
    Compare a specific feature value between extracted and reference DataFrames.
    
    Args:
        extracted_df: DataFrame with extracted features
        reference_df: DataFrame with reference features
        feature_column: Name of the feature column to compare (default: layer20_feat_1)
        
    Returns:
        Dict with comparison results
    """
    print(f"Comparing {feature_column} between extracted and reference features")
    
    # Check if the feature column exists in both DataFrames
    if feature_column not in extracted_df.columns:
        print(f"Error: {feature_column} not found in extracted features")
        print(f"Available columns: {extracted_df.columns.tolist()}")
        return {"match": False, "reason": f"{feature_column} not found in extracted features"}
    
    if feature_column not in reference_df.columns:
        print(f"Error: {feature_column} not found in reference features")
        print(f"Available columns: {reference_df.columns.tolist()}")
        return {"match": False, "reason": f"{feature_column} not found in reference features"}
    
    # Get the specific feature values (first row only for consistency)
    extracted_value = extracted_df[feature_column].values[0]
    reference_value = reference_df[feature_column].values[0]
    
    # Calculate the difference
    abs_diff = abs(extracted_value - reference_value)
    
    print(f"  Extracted value: {extracted_value}")
    print(f"  Reference value: {reference_value}")
    print(f"  Absolute difference: {abs_diff}")
    
    # Determine if the values match (with some tolerance)
    # Using a relative tolerance here
    rel_tol = 1e-4
    abs_tol = 1e-8
    match = abs_diff <= max(rel_tol * max(abs(extracted_value), abs(reference_value)), abs_tol)
    
    return {
        "match": match,
        "extracted_value": extracted_value,
        "reference_value": reference_value,
        "abs_diff": abs_diff
    }

def save_best_config(config, diff):
    """Save the best configuration to a file for future reference."""
    with open("best_feature_config.txt", "w") as f:
        f.write("Best configuration for feature extraction:\n")
        f.write(f"Segmentation Type: {config['seg_type']}\n")
        f.write(f"Alpha: {config['alpha']}\n")
        f.write(f"Extraction Method: {config['extraction_method']}\n")
        f.write(f"Layer Number: {config['layer_num']}\n")
        f.write(f"Normalize Volume: {config['normalize_volume']}\n")
        f.write(f"Normalize Across Volume: {config['normalize_across_volume']}\n")
        f.write(f"Smart Crop: {config['smart_crop']}\n")
        f.write(f"Presynapse Weight: {config['presynapse_weight']}\n")
        f.write(f"Normalize Presynapse Size: {config['normalize_presynapse_size']}\n")
        f.write(f"Target Percentage: {config['target_percentage']}\n")
        f.write(f"Size Tolerance: {config['size_tolerance']}\n")
        f.write(f"Absolute Difference: {diff:.8f}\n")
    print(f"Best configuration saved to best_feature_config.txt")

def main():
    """Main function to run the test."""
    try:
        print("Starting feature extraction test")
        
        # Load reference features from check.xlsx
        reference_df = load_check_file()
        if reference_df is None:
            print("Failed to load reference features. Exiting.")
            return
        
        # Find the specific sample in the reference data
        target_sample = reference_df[reference_df['Var1'] == 'non_spine_synapsed_056']
        if len(target_sample) == 0:
            print("Error: Target sample not found in reference data. Exiting.")
            return
        
        # Load the specific sample (bbox1 non_spine_synapsed_056)
        vol_data_dict, sample_df = load_specific_sample(sample_name="non_spine_synapsed_056", bbox_name="bbox1")
        if vol_data_dict is None or len(sample_df) == 0:
            print("Failed to load specific sample. Exiting.")
            return
        
        # Define fixed parameters
        fixed_params = {
            "seg_type": 10,
            "alpha": 1,
            "extraction_method": "stage_specific",
            "layer_num": 20
        }
        
        # Define variable parameter combinations to test
        variable_param_combinations = [
            # Base combination
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": False,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            
            # Testing normalize_across_volume
            {
                "normalize_volume": False,
                "normalize_across_volume": True,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": False,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            
            # Testing smart_crop
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": True,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": False,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            
            # Testing presynapse_weight variations
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.3,
                "normalize_presynapse_size": False,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.7,
                "normalize_presynapse_size": False,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            
            # Testing normalize_presynapse_size
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": True,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            
            # Testing target_percentage and size_tolerance variations
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": True,
                "target_percentage": 0.2,
                "size_tolerance": 0.1
            },
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": True,
                "target_percentage": 0.4,
                "size_tolerance": 0.1
            },
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": True,
                "target_percentage": None,
                "size_tolerance": 0.05
            },
            {
                "normalize_volume": False,
                "normalize_across_volume": False,
                "smart_crop": False,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": True,
                "target_percentage": None,
                "size_tolerance": 0.2
            },
            
            # Testing combinations
            {
                "normalize_volume": False,
                "normalize_across_volume": True,
                "smart_crop": True,
                "presynapse_weight": 0.5,
                "normalize_presynapse_size": False,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            {
                "normalize_volume": False,
                "normalize_across_volume": True,
                "smart_crop": False,
                "presynapse_weight": 0.3,
                "normalize_presynapse_size": True,
                "target_percentage": None,
                "size_tolerance": 0.1
            },
            {
                "normalize_volume": False,
                "normalize_across_volume": True,
                "smart_crop": True,
                "presynapse_weight": 0.7,
                "normalize_presynapse_size": True,
                "target_percentage": None,
                "size_tolerance": 0.1
            }
        ]
        
        best_match = {"match": False, "config": None, "diff": float('inf')}
        
        for i, var_params in enumerate(variable_param_combinations):
            print("\n" + "="*50)
            print(f"Testing configuration {i+1}/{len(variable_param_combinations)}:")
            
            # Combine fixed and variable parameters
            config = {**fixed_params, **var_params}
            
            # Extract features with current configuration
            extracted_df = extract_features_with_config(vol_data_dict, sample_df, **config)
            
            # Compare specific feature value with reference
            result = compare_feature_value(extracted_df, target_sample, feature_column='layer20_feat_1')
            
            if result.get("match", False):
                print("✅ MATCH FOUND! This configuration matches the reference feature value.")
                best_match = {"match": True, "config": config, "diff": result.get("abs_diff", 0)}
                break
            elif result.get("abs_diff", float('inf')) < best_match["diff"]:
                best_match = {"match": False, "config": config, "diff": result.get("abs_diff", float('inf'))}
        
        print("\n" + "="*50)
        if best_match["match"]:
            print("✅ Success! Found a matching configuration:")
        else:
            print("❌ No exact match found. Best configuration was:")
        
        if best_match["config"]:
            print(f"  Segmentation Type: {best_match['config']['seg_type']}")
            print(f"  Alpha: {best_match['config']['alpha']}")
            print(f"  Extraction Method: {best_match['config']['extraction_method']}")
            print(f"  Layer Number: {best_match['config']['layer_num']}")
            print(f"  Normalize Volume: {best_match['config']['normalize_volume']}")
            print(f"  Normalize Across Volume: {best_match['config']['normalize_across_volume']}")
            print(f"  Smart Crop: {best_match['config']['smart_crop']}")
            print(f"  Presynapse Weight: {best_match['config']['presynapse_weight']}")
            print(f"  Normalize Presynapse Size: {best_match['config']['normalize_presynapse_size']}")
            print(f"  Target Percentage: {best_match['config']['target_percentage']}")
            print(f"  Size Tolerance: {best_match['config']['size_tolerance']}")
            print(f"  Absolute Difference: {best_match['diff']:.8f}")
            
            # Save the best configuration to a file
            save_best_config(best_match['config'], best_match['diff'])
        else:
            print("  No valid configuration was found.")
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 