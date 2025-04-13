"""
Synapse Analysis Pipeline - Feature Extraction

This module provides functionality for extracting features from synapse data
and saving them to CSV format.
"""

import os
import sys
import torch
import pandas as pd
import datetime
from pathlib import Path
import numpy as np
import imageio

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import from newdl module instead of synapse
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from synapse import config
from newdl.dataset3 import SynapseDataset
from newdl.dataloader3 import Synapse3DProcessor, SynapseDataLoader
from inference import (
    extract_and_save_features,
    load_and_prepare_data,
    VGG3D
)

class FeatureExtraction:
    """
    Class to handle synapse feature extraction and saving to CSV.
    """
    
    def __init__(self, config_obj=None):
        """
        Initialize the feature extraction with configuration.
        
        Args:
            config_obj: Configuration object or None to use default config
        """
        self.config = config_obj if config_obj else config
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.processor = None
        self.vol_data_dict = None
        self.syn_df = None
        self.features_df = None
        self.run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_parent_dir = None
        
    def create_results_directory(self):
        """
        Create a timestamped directory for saving results.
        """
        if hasattr(self.config, 'results_dir') and self.config.results_dir:
            results_base_dir = self.config.results_dir
        else:
            if hasattr(self.config, 'csv_output_dir') and 'csv_outputs' in self.config.csv_output_dir:
                results_base_dir = os.path.dirname(self.config.csv_output_dir)
            else:
                workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
                results_base_dir = os.path.join(workspace_dir, "results")
        
        self.results_parent_dir = os.path.join(results_base_dir, f"run_{self.run_timestamp}")
        os.makedirs(self.results_parent_dir, exist_ok=True)
        
        print(f"Created results directory: {self.results_parent_dir}")
        return self.results_parent_dir
        
    def load_data(self):
        """
        Load data using dataloader.py and dataset.py
        """
        print("Loading data...")
        self.vol_data_dict, self.syn_df = load_and_prepare_data(self.config)
        
        # Initialize processor with volume-wide normalization disabled
        self.processor = Synapse3DProcessor(size=self.config.size)
        # Disable normalization for consistent gray values
        self.processor.normalize_volume = False

        self.dataset = SynapseDataset(
            self.vol_data_dict, 
            self.syn_df, 
            processor=self.processor,
            segmentation_type=self.config.segmentation_type,
            alpha=self.config.alpha,
            normalize_across_volume=False  # Set to False for consistent gray values
        )
        
        self.dataloader = SynapseDataLoader(
            raw_base_dir=self.config.raw_base_dir,
            seg_base_dir=self.config.seg_base_dir,
            add_mask_base_dir=self.config.add_mask_base_dir
        )
        
        # Visualize 4 random samples using sample_fig2.py
        print("\nVisualizing 4 random samples to verify data loading...")
        vis_dir = os.path.join(self.results_parent_dir, "sample_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get 4 random indices from the dataset
        indices = torch.randperm(len(self.dataset))[:4].tolist()
        
        for idx in indices:
            # Get the sample
            pixel_values, syn_info, bbox_name = self.dataset[idx]
            print(f"\nProcessing sample {idx}: {syn_info['Var1']} from {bbox_name}")
            
            # Denormalize the cube values using ImageNet parameters
            denormalized_cube = pixel_values * torch.tensor([0.229]) + torch.tensor([0.485])
            denormalized_cube = torch.clamp(denormalized_cube, 0, 1)
            
            # Convert to frames
            frames = denormalized_cube.squeeze(1).numpy()
            
            # Apply global normalization across all frames to maintain consistent gray levels
            # instead of per-frame normalization
            frames_min, frames_max = frames.min(), frames.max()
            if frames_max > frames_min:  # Avoid division by zero
                normalized_frames = (frames - frames_min) / (frames_max - frames_min)
            else:
                normalized_frames = frames
                
            # Convert to uint8 for GIF
            enhanced_frames = [(frame * 255).astype(np.uint8) for frame in normalized_frames]
            
            # Save as GIF
            output_path = os.path.join(vis_dir, f"sample_{idx}_{bbox_name}_{syn_info['Var1']}.gif")
            try:
                print(f"Saving GIF with {len(enhanced_frames)} frames to {output_path}")
                imageio.mimsave(output_path, enhanced_frames, fps=10)
                print(f"GIF saved successfully at {output_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
        
        print("\nSample visualization complete. Check the visualizations in:", vis_dir)
        
        return self.dataset, self.dataloader
    
    def load_model(self):
        """
        Load the VGG3D model
        """
        print("Loading model...")
        self.model = VGG3D()
        return self.model
    
    def extract_features(self, seg_type=None, alpha=None, extraction_method=None, layer_num=None):
        """
        Extract features and save them to CSV.
        
        Args:
            seg_type: Segmentation type to use
            alpha: Alpha value for feature extraction
            extraction_method: Method to extract features ('standard' or 'stage_specific')
            layer_num: Layer number to extract features from if using stage_specific method
        """
        print("Extracting features...")
        
        if extraction_method is None:
            extraction_method = getattr(self.config, 'extraction_method', 'standard')
        
        if layer_num is None and extraction_method == 'stage_specific':
            layer_num = getattr(self.config, 'layer_num', 20)
            
        # Create directory name with all parameters
        dir_components = [
            "features",
            f"extraction_{extraction_method}",
            f"layer{layer_num}" if extraction_method == 'stage_specific' else None,
            f"seg{seg_type}",
            f"alpha{alpha}",
        ]
        # Filter out None values and join with underscores
        output_dir_name = "_".join(filter(None, dir_components))
        output_dir = os.path.join(self.results_parent_dir, output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Using {extraction_method} feature extraction method")
        if extraction_method == 'stage_specific':
            print(f"Extracting features from layer {layer_num}")

        features_path = extract_and_save_features(
            self.model, 
            self.dataset, 
            self.config, 
            seg_type, 
            alpha, 
            output_dir,
            extraction_method=extraction_method,
            layer_num=layer_num,
            perform_clustering=False  # Disable clustering
        )
        
        if isinstance(features_path, str):
            if os.path.exists(features_path):
                self.features_df = pd.read_csv(features_path)
                print(f"Loaded features from {features_path}, shape: {self.features_df.shape}")
            else:
                raise FileNotFoundError(f"Features file not found at {features_path}")
        else:
            self.features_df = features_path
            
        return self.features_df

def main():
    """Main function to run feature extraction"""
    config.parse_args()
    
    if not hasattr(config, 'results_dir') or not config.results_dir:
        if hasattr(config, 'csv_output_dir') and 'csv_outputs' in config.csv_output_dir:
            config.results_dir = os.path.dirname(config.csv_output_dir)
        else:
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            config.results_dir = os.path.join(workspace_dir, "results")
    
    # Get feature extraction parameters from config
    seg_type = config.segmentation_type
    alpha = config.alpha
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    
    print(f"Results will be saved in: {config.results_dir}/run_TIMESTAMP")
    
    # Create and run feature extraction
    extractor = FeatureExtraction(config)
    extractor.create_results_directory()
    extractor.load_data()
    extractor.load_model()
    features_df = extractor.extract_features(
        seg_type=seg_type,
        alpha=alpha,
        extraction_method=extraction_method,
        layer_num=layer_num
    )
    
    print(f"Features extracted and saved to: {extractor.results_parent_dir}")

if __name__ == "__main__":
    main() 