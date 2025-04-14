"""
Synapse Analysis Pipeline - High Level Orchestrator

This module provides a high-level orchestrator for the synapse analysis pipeline,
defining stages for data loading, model initialization, feature extraction,
clustering, and visualization without implementing the actual functionality.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import datetime
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multi_layer_cam import visualize_cluster_attention

# Import from newdl module instead of synapse
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
# Import necessary modules without implementing their functionality
from synapse import config
from newdl.dataloader3 import SynapseDataLoader
from inference import (
    extract_features, 
    create_plots, 
    extract_and_save_features,
    run_full_analysis,
    load_and_prepare_data,
    run_clustering_analysis,
    create_sample_visualizations,
    find_random_samples_in_clusters,
    save_cluster_samples,
    apply_umap,
    VGG3D
)
from multi_layer_cam import SimpleGradCAM, process_single_sample
from presynapse_analysis import (
    load_feature_data,
    identify_synapses_with_same_presynapse, 
    create_bbox_colored_umap,
    create_cluster_colored_umap,
    create_cluster_visualizations,
    run_presynapse_analysis
)


class SynapsePipeline:
    """
    High-level class to orchestrate the synapse analysis pipeline.
    This class defines the stages of the pipeline without implementing
    the actual functionality, delegating to the appropriate modules.
    """
    
    def __init__(self, config_obj=None):
        """
        Initialize the pipeline with configuration.
        
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
        Create a timestamped parent directory for all results from this run.
        """
        # First, try to extract the results base directory from config
        # Assuming config might have a results_dir attribute or similar
        if hasattr(self.config, 'results_dir') and self.config.results_dir:
            results_base_dir = self.config.results_dir
        else:
            # Check if csv_output_dir has 'csv_outputs' in it, and use its parent if so
            if hasattr(self.config, 'csv_output_dir') and 'csv_outputs' in self.config.csv_output_dir:
                results_base_dir = os.path.dirname(self.config.csv_output_dir)
            else:
                # Fallback to using workspace path
                workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
                results_base_dir = os.path.join(workspace_dir, "results")
        
        # Create parent directory with timestamp directly under results base
        self.results_parent_dir = os.path.join(results_base_dir, f"run_{self.run_timestamp}")
        os.makedirs(self.results_parent_dir, exist_ok=True)
        
        print(f"Created results directory: {self.results_parent_dir}")
        return self.results_parent_dir
        
    def load_data(self):
        """
        Load data using dataloader.py and dataset.py
        Creates the dataset and dataloader instances
        """
        # Load and prepare data (delegate to existing functions)
        print("Loading data...")
        self.vol_data_dict, self.syn_df = load_and_prepare_data(self.config)
        
        # Initialize processor
        self.processor = Synapse3DProcessor(size=self.config.size)
        # Disable normalization for consistent gray values
        self.processor.normalize_volume = False
        
        # Create dataset with segmentation_type parameter
        self.dataset = SynapseDataset(
            self.vol_data_dict, 
            self.syn_df, 
            processor=self.processor,
            segmentation_type=self.config.segmentation_type,
            alpha=self.config.alpha,
            normalize_across_volume=False  # Set to False for consistent gray values
        )
        
        # Create dataloader if needed for batch processing
        self.dataloader = SynapseDataLoader(
            raw_base_dir=self.config.raw_base_dir,
            seg_base_dir=self.config.seg_base_dir,
            add_mask_base_dir=self.config.add_mask_base_dir
        )
        
        return self.dataset, self.dataloader
    
    def load_model(self):
        """
        Load model from model folder
        """
        print("Loading model...")
        self.model = VGG3D()
        return self.model
    
    def extract_features(self, seg_type=None, alpha=None, extraction_method=None, layer_num=None, pooling_method=None):
        """
        Extract features using inference.py
        
        Args:
            seg_type: Segmentation type to use
            alpha: Alpha value for feature extraction
            extraction_method: Method to extract features ('standard' or 'stage_specific')
            layer_num: Layer number to extract features from if using stage_specific method
            pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        """
        print("Extracting features...")
        
        # Use config values if parameters not provided
        if extraction_method is None:
            extraction_method = getattr(self.config, 'extraction_method', 'standard')
        
        if layer_num is None and extraction_method == 'stage_specific':
            layer_num = getattr(self.config, 'layer_num', 20)
        
        if pooling_method is None:
            pooling_method = getattr(self.config, 'pooling_method', 'avg')
            
        # Create appropriate output directory name based on extraction method
        if extraction_method == 'stage_specific':
            output_dir = os.path.join(self.results_parent_dir, f"features_layer{layer_num}_{pooling_method}_seg{seg_type}_alpha{alpha}")
        else:
            output_dir = os.path.join(self.results_parent_dir, f"features_{pooling_method}_seg{seg_type}_alpha{alpha}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Using {extraction_method} feature extraction method with {pooling_method} pooling")
        if extraction_method == 'stage_specific':
            print(f"Extracting features from layer {layer_num}")
        
        # Extract features using the existing function with the new parameters
        features_path = extract_and_save_features(
            self.model, 
            self.dataset, 
            self.config, 
            seg_type, 
            alpha, 
            output_dir,
            extraction_method=extraction_method,
            layer_num=layer_num,
            pooling_method=pooling_method
        )
        
        # Make sure we have a DataFrame, not just a path
        if isinstance(features_path, str):
            # Load the features from the saved CSV file
            if os.path.exists(features_path):
                self.features_df = pd.read_csv(features_path)
                print(f"Loaded features from {features_path}, shape: {self.features_df.shape}")
            else:
                raise FileNotFoundError(f"Features file not found at {features_path}")
        else:
            # If it's already a DataFrame, just use it
            self.features_df = features_path
            
        return self.features_df
    
    def cluster_features(self, output_dir=None):
        """
        Cluster the extracted features
        
        Args:
            output_dir: Directory to save clustering results
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_parent_dir, "clustering_results")
        os.makedirs(output_dir, exist_ok=True)
        
        print("Clustering features...")
        result = run_clustering_analysis(self.features_df, output_dir)
        
        # Handle the result which might be a path to a CSV or a DataFrame
        if isinstance(result, str):
            # Load the clustered features from the saved CSV file
            if os.path.exists(result):
                self.features_df = pd.read_csv(result)
                print(f"Loaded clustered features from {result}, shape: {self.features_df.shape}")
            else:
                raise FileNotFoundError(f"Clustered features file not found at {result}")
        else:
            # If it's already a DataFrame, just use it
            self.features_df = result
            
        return self.features_df
    
    def create_dimension_reduction_visualizations(self, output_dir=None):
        """
        Create dimensionality reduction visualizations (UMAP)
        - One colored by bbox number
        - One colored by cluster number
        
        Args:
            output_dir: Directory to save visualization results
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_parent_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating dimension reduction visualizations...")
        
        # Create UMAP visualizations
        create_bbox_colored_umap(self.features_df, output_dir)
        create_cluster_colored_umap(self.features_df, output_dir)
        
        return output_dir
    
    def create_cluster_sample_visualizations(self, num_samples=4, attention_layer=20, output_dir=None):
        """
        Show samples from each cluster with attention maps
        
        Args:
            num_samples: Number of samples to show from each cluster
            attention_layer: Layer to use for attention map visualization
            output_dir: Directory to save visualization results
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_parent_dir, "sample_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating sample visualizations with attention maps for layer {attention_layer}...")
        
        # Find samples in clusters - now returns indices directly
        feature_cols = [col for col in self.features_df.columns if col.startswith('feat_')]
        random_samples = find_random_samples_in_clusters(
            self.features_df, 
            feature_cols, 
            n_samples=num_samples
        )
        
        # Save regular cluster samples (without attention)
        save_cluster_samples(self.dataset, random_samples, output_dir)
        
        # Generate attention visualizations using the multi_layer_cam module
        # This will create visualizations with attention overlays for each sample
        attention_dir = os.path.join(output_dir, "attention_maps")
        os.makedirs(attention_dir, exist_ok=True)
        
        # Use the specialized function from multi_layer_cam.py for cluster attention visualization
        
        visualization_results = visualize_cluster_attention(
            self.model,
            self.dataset,
            random_samples,
            attention_dir,
            [attention_layer],  # Just the requested layer
            [f"layer{attention_layer}"],
            n_slices=3  # Show 3 slices per sample
        )
        
        print(f"Attention visualizations saved to {attention_dir}")
        
        return random_samples
    
    def analyze_bounding_boxes_in_clusters(self):
        """
        Analyze and visualize the distribution of bounding boxes in clusters
        """
        print("Analyzing bounding boxes in clusters...")
        
        # Call the function provided by the user
        cluster_counts = self.count_bboxes_in_clusters(self.features_df)
        self.plot_bboxes_in_clusters(cluster_counts)
        
        return cluster_counts
    
    def count_bboxes_in_clusters(self, features_df):
        """
        Count the occurrences of each bounding box in each cluster.
        Returns a DataFrame with the counts.
        """
        # Create a pivot table where rows are clusters and columns are bounding boxes
        cluster_counts = features_df.groupby(['cluster', 'bbox_name']).size().unstack(fill_value=0)
        return cluster_counts
    
    def plot_bboxes_in_clusters(self, cluster_counts):
        """
        Plot various visualizations showing the count of each bounding box 
        in each cluster using Plotly.
        """
        # This function would delegate to the user-provided code
        # The implementation would be based on the code snippet provided by the user
        print("Plotting bounding box distributions in clusters...")
        
    def run_presynapse_analysis(self, output_dir=None):
        """
        Run presynapse analysis and create summary visualizations
        
        Args:
            output_dir: Directory to save analysis results
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_parent_dir, "presynapse_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running presynapse analysis...")
        
        # Save the original output directories
        orig_csv_output_dir = self.config.csv_output_dir
        orig_clustering_results = getattr(self.config, 'clustering_results_dir', None)
        orig_clustering_output_dir = getattr(self.config, 'clustering_output_dir', None)
        
        # Temporarily set the directories to use our timestamped parent directory
        self.config.csv_output_dir = self.results_parent_dir
        if hasattr(self.config, 'clustering_results_dir'):
            self.config.clustering_results_dir = os.path.join(self.results_parent_dir, "clustering_results")
        
        # This is the key parameter used by presynapse_analysis.py
        if hasattr(self.config, 'clustering_output_dir'):
            print(f"Setting clustering_output_dir to {self.results_parent_dir}")
            self.config.clustering_output_dir = self.results_parent_dir
        
        # Run the presynapse analysis with the updated paths
        run_presynapse_analysis(self.config)
        
        # Restore the original output directories
        self.config.csv_output_dir = orig_csv_output_dir
        if orig_clustering_results is not None:
            self.config.clustering_results_dir = orig_clustering_results
        if orig_clustering_output_dir is not None:
            self.config.clustering_output_dir = orig_clustering_output_dir
        
        return output_dir
    
    def run_full_pipeline(self, seg_type=1, alpha=1, num_samples=4, attention_layer=20, extraction_method=None, layer_num=None, pooling_method=None):
        """
        Run the full analysis pipeline from data loading to visualization
        
        Args:
            seg_type: Segmentation type to use
            alpha: Alpha value for feature extraction
            num_samples: Number of samples to show from each cluster
            attention_layer: Layer to use for attention map visualization
            extraction_method: Method to extract features ('standard' or 'stage_specific')
            layer_num: Layer number to extract features from if using stage_specific method
            pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        """
        print(f"Running full pipeline with seg_type={seg_type}, alpha={alpha}")
        
        # Use config values if parameters not provided
        if extraction_method is None:
            extraction_method = getattr(self.config, 'extraction_method', 'standard')
        
        if layer_num is None and extraction_method == 'stage_specific':
            layer_num = getattr(self.config, 'layer_num', 20)
            
        if pooling_method is None:
            pooling_method = getattr(self.config, 'pooling_method', 'avg')
            
        if extraction_method == 'stage_specific':
            print(f"Using stage-specific feature extraction from layer {layer_num}")
        else:
            print("Using standard feature extraction")
            
        print(f"Using {pooling_method} pooling method")
        
        # Create the base output directory if it doesn't exist
        os.makedirs(self.config.csv_output_dir, exist_ok=True)
        os.makedirs(self.config.save_gifs_dir, exist_ok=True)
        
        # Create timestamped parent directory for all results
        self.create_results_directory()
        
        # Save original directory values
        original_dirs = {
            'gifs_dir': self.config.save_gifs_dir,
            'csv_output_dir': self.config.csv_output_dir,
            'clustering_results_dir': getattr(self.config, 'clustering_results_dir', None),
            'clustering_results_final': getattr(self.config, 'clustering_results_final', None),
            'clustering_output_dir': getattr(self.config, 'clustering_output_dir', None)
        }
        
        # Update directories to use our timestamped parent directory
        timestamped_gifs_dir = os.path.join(self.results_parent_dir, "gifs")
        os.makedirs(timestamped_gifs_dir, exist_ok=True)
        self.config.save_gifs_dir = timestamped_gifs_dir
        
        # Handle clustering_results_final directory if it's configured
        if hasattr(self.config, 'clustering_results_final'):
            clustering_results_final_dir = os.path.join(self.results_parent_dir, "clustering_results_final")
            os.makedirs(clustering_results_final_dir, exist_ok=True)
            self.config.clustering_results_final = clustering_results_final_dir
        
        # Also handle clustering_results_dir if it's configured
        if hasattr(self.config, 'clustering_results_dir'):
            clustering_results_dir = os.path.join(self.results_parent_dir, "clustering_results")
            os.makedirs(clustering_results_dir, exist_ok=True)
            self.config.clustering_results_dir = clustering_results_dir
            
        # Handle clustering_output_dir used by presynapse_analysis
        if hasattr(self.config, 'clustering_output_dir'):
            print(f"Setting clustering_output_dir to {self.results_parent_dir}")
            self.config.clustering_output_dir = self.results_parent_dir
        
        try:
            # Run each stage of the pipeline
            self.load_data()
            self.load_model()
            self.extract_features(seg_type, alpha, extraction_method, layer_num, pooling_method)
            self.cluster_features()
            self.create_dimension_reduction_visualizations()
            # self.create_cluster_sample_visualizations(num_samples, attention_layer)
            # self.analyze_bounding_boxes_in_clusters()
            # self.run_presynapse_analysis()
            
            print(f"Pipeline completed successfully! All results are in: {self.results_parent_dir}")
            
        finally:
            # Restore all original directory values
            self.config.save_gifs_dir = original_dirs['gifs_dir']
            self.config.csv_output_dir = original_dirs['csv_output_dir']
            
            if original_dirs['clustering_results_dir'] is not None:
                self.config.clustering_results_dir = original_dirs['clustering_results_dir']
            
            if original_dirs['clustering_results_final'] is not None:
                self.config.clustering_results_final = original_dirs['clustering_results_final']
                
            if original_dirs['clustering_output_dir'] is not None:
                self.config.clustering_output_dir = original_dirs['clustering_output_dir']
        
        return {
            "features_df": self.features_df,
            "model": self.model,
            "dataset": self.dataset,
            "results_dir": self.results_parent_dir
        }


def main():
    """Main function to run the pipeline"""
    # Parse command line arguments
    config.parse_args()
    
    # Set results_dir if not already set (for consistency with the new structure)
    if not hasattr(config, 'results_dir') or not config.results_dir:
        if hasattr(config, 'csv_output_dir') and 'csv_outputs' in config.csv_output_dir:
            # If csv_output_dir contains 'csv_outputs', use its parent directory
            config.results_dir = os.path.dirname(config.csv_output_dir)
        else:
            # Otherwise, try to find the results directory
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            config.results_dir = os.path.join(workspace_dir, "results")
    
    # Get feature extraction parameters from config
    seg_type = config.segmentation_type
    alpha = config.alpha
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    
    print(f"Results will be saved in: {config.results_dir}/run_TIMESTAMP")
    
    # Create and run the pipeline
    pipeline = SynapsePipeline(config)
    results = pipeline.run_full_pipeline(
        seg_type=seg_type,
        alpha=alpha,
        extraction_method=extraction_method,
        layer_num=layer_num
    )
    
    print(f"Results saved to: {results['results_dir']}")


if __name__ == "__main__":
    main() 