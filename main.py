#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SynapseClusterEM Main Script
============================

This script serves as the main entry point for the SynapseClusterEM project,
providing a comprehensive workflow for analyzing and clustering 3D synapse
structures from electron microscopy (EM) data.

Usage:
    python main.py --mode [preprocess|extract|cluster|visualize|all]
    python main.py --config config.yaml
"""

import os
import sys
import argparse
import json
import logging
import yaml
from pathlib import Path
import pandas as pd
import torch
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import glob

# Import project modules
from synapse_analysis.models.vgg3d import Vgg3D, load_model_from_checkpoint
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data,
    calculate_global_stats
)
from synapse_analysis.data.dataset import SynapseDataset
from synapse_analysis.analysis.feature_extraction import extract_and_save_features
from synapse_analysis.analysis.clustering import (
    perform_clustering,
    compute_embeddings,
    analyze_clusters,
    save_cluster_visualizations
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('SynapseClusterEM')

def setup_file_logger(output_dir):
    """Set up file logging in addition to console logging"""
    log_file = os.path.join(output_dir, f"synapse_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return log_file

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        sys.exit(1)

def config_to_args(config):
    """Convert configuration dictionary to argparse namespace"""
    args = argparse.Namespace()
    
    # Workflow control
    args.mode = config.get('mode', 'all')
    
    # Data paths
    data_paths = config.get('data_paths', {})
    args.raw_base_dir = data_paths.get('raw_base_dir')
    args.seg_base_dir = data_paths.get('seg_base_dir')
    args.add_mask_base_dir = data_paths.get('add_mask_base_dir', '')
    args.excel_dir = data_paths.get('excel_dir')
    args.output_dir = data_paths.get('output_dir', 'outputs/default')
    args.checkpoint_path = data_paths.get('checkpoint_path')
    
    # Dataset parameters
    dataset = config.get('dataset', {})
    args.bbox_names = dataset.get('bbox_names', ['bbox1'])
    args.size = dataset.get('size', [80, 80])
    args.subvol_size = dataset.get('subvol_size', 80)
    args.num_frames = dataset.get('num_frames', 80)
    
    # Analysis parameters
    analysis = config.get('analysis', {})
    args.segmentation_types = analysis.get('segmentation_types', [9, 10])
    args.alphas = analysis.get('alphas', [1.0])
    args.n_clusters = analysis.get('n_clusters', 10)
    args.clustering_method = analysis.get('clustering_method', 'kmeans')
    args.batch_size = analysis.get('batch_size', 2)
    args.num_workers = analysis.get('num_workers', 0)
    
    # Global normalization parameters
    normalization = config.get('normalization', {})
    args.use_global_norm = normalization.get('use_global_norm', False)
    args.global_stats_path = normalization.get('global_stats_path')
    args.num_samples_for_stats = normalization.get('num_samples_for_stats', 100)
    
    # Visualization parameters
    visualization = config.get('visualization', {})
    args.create_3d_plots = visualization.get('create_3d_plots', False)
    args.save_interactive = visualization.get('save_interactive', False)
    
    # System parameters
    system = config.get('system', {})
    args.gpu_id = system.get('gpu_id', 0)
    args.seed = system.get('seed', 42)
    args.verbose = system.get('verbose', True)
    
    return args

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SynapseClusterEM Analysis Pipeline")
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    
    # Workflow control
    parser.add_argument('--mode', type=str, choices=['preprocess', 'extract', 'cluster', 'visualize', 'all'],
                        default='all', help='Pipeline mode to run')
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, help='Directory containing raw image data')
    parser.add_argument('--seg_base_dir', type=str, help='Directory containing segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, default='', help='Directory containing additional mask data')
    parser.add_argument('--excel_dir', type=str, help='Directory containing Excel files with synapse information')
    parser.add_argument('--output_dir', type=str, default='outputs/main_results', help='Directory to save output files')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the VGG3D model checkpoint')
    
    # Dataset parameters
    parser.add_argument('--bbox_names', type=str, nargs='+', default=['bbox1'], 
                        help='Names of bounding boxes to process')
    parser.add_argument('--size', type=int, nargs=2, default=[80, 80], 
                        help='Size of the 2D slices (height, width)')
    parser.add_argument('--subvol_size', type=int, default=80, 
                        help='Size of the 3D subvolume')
    parser.add_argument('--num_frames', type=int, default=80, 
                        help='Number of frames in the 3D volume')
    
    # Analysis parameters
    parser.add_argument('--segmentation_types', type=int, nargs='+', default=[9, 10], 
                        help='Segmentation types to analyze')
    parser.add_argument('--alphas', type=float, nargs='+', default=[1.0], 
                        help='Alpha values for blending')
    parser.add_argument('--n_clusters', type=int, default=10, 
                        help='Number of clusters for K-means')
    parser.add_argument('--clustering_method', type=str, choices=['kmeans', 'dbscan'], default='kmeans',
                        help='Clustering method to use')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers for data loading')
    
    # Global normalization parameters
    parser.add_argument('--use_global_norm', action='store_true', 
                        help='Use global normalization')
    parser.add_argument('--global_stats_path', type=str, 
                        help='Path to saved global stats JSON (will calculate if not provided)')
    parser.add_argument('--num_samples_for_stats', type=int, default=100, 
                        help='Number of samples for global stats (0 for all)')
    
    # Visualization parameters
    parser.add_argument('--create_3d_plots', action='store_true', 
                        help='Create 3D visualizations')
    parser.add_argument('--save_interactive', action='store_true', 
                        help='Save interactive HTML visualizations')
    
    # System parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # If config file is provided, load it and override with command line arguments
    if args.config:
        config = load_config(args.config)
        config_args = config_to_args(config)
        
        # Override config values with command line arguments if provided
        for key, value in vars(args).items():
            if key != 'config' and value is not None:
                # For boolean flags that default to False
                if isinstance(value, bool) and value:
                    setattr(config_args, key, value)
                # For other arguments that were explicitly provided AND not empty strings
                elif not (isinstance(value, (bool, list)) and value == parser.get_default(key)) and not (isinstance(value, str) and value == ''):
                    setattr(config_args, key, value)
        
        args = config_args
    
    # Validate required arguments
    required_args = ['raw_base_dir', 'seg_base_dir', 'excel_dir', 'checkpoint_path']
    missing_args = [arg for arg in required_args if getattr(args, arg, None) is None]
    
    if missing_args and args.mode not in ['cluster', 'visualize']:
        logger.error(f"Missing required arguments: {', '.join(missing_args)}")
        parser.print_help()
        sys.exit(1)
    
    return args

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def preprocess_data(args):
    """Preprocess data and calculate global normalization statistics if needed"""
    logger.info("Starting data preprocessing...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If global normalization is requested but no stats file is provided, calculate them
    if args.use_global_norm and not args.global_stats_path:
        # Check if global stats file already exists in the output directory
        global_stats_path = os.path.join(args.output_dir, 'global_stats.json')
        
        if os.path.exists(global_stats_path):
            logger.info(f"Global statistics file {global_stats_path} already exists. Skipping calculation.")
            args.global_stats_path = global_stats_path
        else:
            logger.info("Calculating global normalization statistics...")
            
            # Calculate global stats
            global_stats = calculate_global_stats(
                raw_base_dir=args.raw_base_dir,
                seg_base_dir=args.seg_base_dir,
                add_mask_base_dir=args.add_mask_base_dir,
                excel_dir=args.excel_dir,
                segmentation_types=args.segmentation_types,
                bbox_names=args.bbox_names,
                num_samples=args.num_samples_for_stats
            )
            
            # Save global stats
            with open(global_stats_path, 'w') as f:
                json.dump(global_stats, f)
            
            args.global_stats_path = global_stats_path
            logger.info(f"Global statistics saved to {global_stats_path}")
    
    return args

def extract_features(args):
    """Extract features from synapse volumes"""
    logger.info("Starting feature extraction...")
    
    # Set device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        logger.info(f"gpu_id parameter value: {args.gpu_id}")
    
    # Load global normalization statistics if available
    if args.use_global_norm:
        if args.global_stats_path:
            global_stats_path = args.global_stats_path
        else:
            global_stats_path = os.path.join(args.output_dir, "global_stats.json")
        
        if os.path.exists(global_stats_path):
            with open(global_stats_path, 'r') as f:
                global_stats = json.load(f)
            logger.info(f"Loaded global statistics from {global_stats_path}")
        else:
            logger.warning(f"Global statistics file not found at {global_stats_path}")
            global_stats = {"mean": 0.0, "std": 1.0}
    else:
        global_stats = None
    
    # Process each segmentation type
    all_features = []
    
    # Debug print for Excel directory
    print(f"Excel directory: {args.excel_dir}")
    print(f"Excel directory exists: {os.path.exists(args.excel_dir)}")
    for bbox in args.bbox_names:
        excel_path = os.path.join(args.excel_dir, f"{bbox}.xlsx")
        print(f"Looking for Excel file: {excel_path}")
        print(f"Excel file exists: {os.path.exists(excel_path)}")
    
    # Load synapse data
    try:
        synapse_data = load_synapse_data(
            args.bbox_names,
            args.excel_dir
        )
    except Exception as e:
        logger.error(f"Error loading synapse data: {e}")
        raise
    
    # Load volumes
    volumes = load_all_volumes(
        bbox_names=args.bbox_names,
        raw_base_dir=args.raw_base_dir,
        seg_base_dir=args.seg_base_dir,
        add_mask_base_dir=args.add_mask_base_dir
    )
    
    # Create a processor
    processor = Synapse3DProcessor(
        size=tuple(args.size),
        apply_global_norm=args.use_global_norm,
        global_stats=global_stats
    )
    
    # Create dataset
    dataset = SynapseDataset(
        vol_data_dict=volumes,
        synapse_df=synapse_data,
        processor=processor,
        segmentation_type=args.segmentation_types[0],
        subvol_size=args.subvol_size,
        alpha=args.alphas[0]  # Pass alpha from config
    )
    
    # Create the model instance first
    model = Vgg3D()
    # Then load the checkpoint
    model = load_model_from_checkpoint(model, args.checkpoint_path)
    model.eval()
    
    # Extract features for each segmentation type
    features_df_list = []
    
    for seg_type in args.segmentation_types:
        logger.info(f"Processing segmentation type {seg_type}...")
        
        # Create output directory for this segmentation type
        seg_output_dir = os.path.join(args.output_dir, f"seg_type_{seg_type}")
        os.makedirs(seg_output_dir, exist_ok=True)
        
        # Check if feature CSV already exists
        norm_suffix = "_global_norm" if args.use_global_norm else ""
        csv_filename = f"features_seg{seg_type}_alpha{str(args.alphas[0]).replace('.', '_')}{norm_suffix}.csv"
        csv_filepath = os.path.join(seg_output_dir, csv_filename)
        
        if os.path.exists(csv_filepath):
            logger.info(f"Feature file {csv_filepath} already exists. Skipping feature extraction.")
            # Load the existing features
            features_df = pd.read_csv(csv_filepath)
        else:
            # Load synapse data
            synapse_data = load_synapse_data(
                excel_dir=args.excel_dir,
                bbox_names=args.bbox_names
            )
            
            # Load volumes
            volumes = load_all_volumes(
                bbox_names=args.bbox_names,
                raw_base_dir=args.raw_base_dir,
                seg_base_dir=args.seg_base_dir,
                add_mask_base_dir=args.add_mask_base_dir
            )
            
            # Create a processor
            processor = Synapse3DProcessor(
                size=tuple(args.size),
                apply_global_norm=args.use_global_norm,
                global_stats=global_stats
            )
            
            # Create dataset
            dataset = SynapseDataset(
                vol_data_dict=volumes,
                synapse_df=synapse_data,
                processor=processor,
                segmentation_type=seg_type,
                subvol_size=args.subvol_size,
                alpha=args.alphas[0]  # Pass alpha from config
            )
            # Visualize sample center slices to verify model input
            def visualize_sample_slices(dataset, num_samples=5, output_dir=None):
                """Visualize center slices of a few samples to verify model input"""
                if num_samples > len(dataset):
                    num_samples = len(dataset)
                    
                # Create output directory if needed
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Get a few samples
                indices = np.linspace(0, len(dataset)-1, num_samples).astype(int)
                
                for i, idx in enumerate(indices):
                    volume, syn_info, bbox_name = dataset[idx]
                    # Shape: [C, D, H, W]
                    synapse_id = syn_info['Var1'] if 'Var1' in syn_info else idx
                    
                    # Get center slice
                    center_slice_idx = volume.shape[1] // 2
                    center_slice = volume[:, center_slice_idx, :, :]  # Shape: [C, H, W]
                    
                    # Calculate appropriate min and max values for display
                    vmin = center_slice.min()
                    vmax = center_slice.max()
                    
                    # Create a figure with subplots for each channel
                    num_channels = center_slice.shape[0]
                    fig_width = 20
                    fig_height = 5
                    
                    fig, axes = plt.subplots(1, num_channels, figsize=(fig_width, fig_height))
                    
                    # Ensure axes is always a list-like object
                    if num_channels == 1:
                        axes = [axes]
                        
                    # Plot each channel with enhanced contrast
                    for c in range(num_channels):
                        # Get the slice data for this channel
                        slice_data = center_slice[c]
                        
                        # Plot the image with appropriate value range
                        axes[c].imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
                        axes[c].set_title(f"Channel {c} - min={vmin:.2f}, max={vmax:.2f}")
                        axes[c].axis('off')
                    
                    # Set a title for the entire figure
                    plt.suptitle(f"Sample {i+1} - bbox: {bbox_name}, id: {synapse_id}")
                    plt.tight_layout()
                    
                    # Save or show the figure
                    if output_dir:
                        filename = f"sample_{i+1}_bbox_{bbox_name}_id_{synapse_id}.png"
                        filepath = os.path.join(output_dir, filename)
                        plt.savefig(filepath, dpi=100, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.show()
                    
                if output_dir:
                    logger.info(f"Sample visualizations saved to {output_dir}")
            
            # Call the visualization function
            sample_viz_dir = visualize_sample_slices(
                dataset=dataset,
                num_samples=5,
                output_dir=os.path.join(seg_output_dir, "sample_visualizations")
            )
            logger.info(f"Sample visualizations created at {sample_viz_dir}")
            
            # Extract features
            csv_filepath = extract_and_save_features(
                model=model,
                dataset=dataset,
                seg_type=seg_type,
                alpha=args.alphas[0],  # Using the first alpha value
                output_dir=seg_output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                apply_global_norm=args.use_global_norm,
                global_stats=global_stats
            )
            
            # Load the saved features from CSV
            features_df = pd.read_csv(csv_filepath)
        
        # Add segmentation type to features
        features_df['segmentation_type'] = seg_type
        features_df_list.append(features_df)
    
    # Combine features from all segmentation types
    if len(features_df_list) > 0:
        combined_features_df = pd.concat(features_df_list, ignore_index=True)
        combined_features_path = os.path.join(args.output_dir, 'combined_features.csv')
        
        # Check if combined features file already exists
        if not os.path.exists(combined_features_path):
            combined_features_df.to_csv(combined_features_path, index=False)
            logger.info(f"Combined features saved to {combined_features_path}")
        else:
            logger.info(f"Combined features file {combined_features_path} already exists. Skipping save.")
        
        return combined_features_df
    else:
        logger.warning("No features were extracted!")
        return None

def perform_cluster_analysis(args, features_df=None):
    """Perform clustering analysis on extracted features"""
    logger.info("Starting clustering analysis...")
    
    # Load features if not provided
    if features_df is None:
        features_path = os.path.join(args.output_dir, 'combined_features.csv')
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            logger.info(f"Loaded features from {features_path}")
        else:
            logger.error(f"Features file not found at {features_path}")
            return None
    
    # Create output directory for clustering results
    cluster_output_dir = os.path.join(args.output_dir, 'clustering')
    os.makedirs(cluster_output_dir, exist_ok=True)
    
    # Extract feature columns (those starting with 'feat_')
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    
    if not feature_cols:
        logger.error("No feature columns found in the DataFrame")
        return None
    
    # Step 1: Perform clustering on the ORIGINAL features only
    logger.info(f"Performing {args.clustering_method} clustering with {args.n_clusters} clusters on extracted features...")
    
    # Create a copy of the features DataFrame to work with
    clustered_df = features_df.copy()
    
    # Apply clustering with specified method
    clustered_df, clustering_model = perform_clustering(
        features_df,
        method=args.clustering_method,
        n_clusters=args.n_clusters
    )
    
    # Check if we got meaningful clusters (more than 1)
    if clustered_df['cluster'].nunique() <= 1:
        logger.warning(f"{args.clustering_method} with {args.n_clusters} clusters only found a single cluster. Trying with fewer clusters...")
        
        # Try with fewer clusters
        for n_clusters in [5, 3, 2]:
            logger.info(f"Trying {args.clustering_method} with {n_clusters} clusters...")
            
            if args.clustering_method == 'kmeans':
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Scale features
                X = features_df[feature_cols].values
                X_scaled = StandardScaler().fit_transform(X)
                
                # Apply KMeans with fewer clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clustered_df['cluster'] = kmeans.fit_predict(X_scaled)
                clustering_model = kmeans
                
                if clustered_df['cluster'].nunique() > 1:
                    logger.info(f"Successfully found {clustered_df['cluster'].nunique()} clusters with KMeans (n_clusters={n_clusters})")
                    break
    
    # If we still only have 1 cluster, create artificial clusters for visualization purposes only
    if clustered_df['cluster'].nunique() <= 1:
        logger.warning("Could not find natural clusters in the data. Creating artificial clusters for visualization...")
        
        # Create evenly sized artificial clusters
        n_samples = len(clustered_df)
        cluster_size = n_samples // 4  # Create 4 roughly equal clusters
        
        clustered_df['cluster'] = 0
        for i in range(1, 4):
            clustered_df.loc[i*cluster_size:(i+1)*cluster_size-1, 'cluster'] = i
            
        logger.info(f"Created 4 artificial clusters for visualization purposes")
    
    # Step 2: Compute UMAP embeddings for visualization only
    logger.info("Computing UMAP embeddings for visualization only...")
    embeddings_2d = compute_embeddings(
        features_df,
        n_components=2,
        method='umap'
    )
    
    # Add 2D embeddings to DataFrame
    clustered_df['umap_x'] = embeddings_2d[:, 0]
    clustered_df['umap_y'] = embeddings_2d[:, 1]
    
    # Compute 3D embeddings if requested
    if args.create_3d_plots:
        logger.info("Computing UMAP embeddings for 3D visualization...")
        embeddings_3d = compute_embeddings(
            features_df,
            n_components=3,
            method='umap'
        )
        clustered_df['umap_z'] = embeddings_3d[:, 2]
    
    # Display clustering results
    logger.info(f"Clustering completed: {clustered_df['cluster'].nunique()} unique clusters found")
    
    # Save clustered data with embeddings
    clustered_path = os.path.join(cluster_output_dir, 'clustered_data.csv')
    clustered_df.to_csv(clustered_path, index=False)
    logger.info(f"Clustered data saved to {clustered_path}")
    
    # Analyze clusters
    logger.info("Analyzing clusters...")
    analyze_clusters(
        clustered_df,
        clustering_model,
        feature_cols,
        output_dir=Path(cluster_output_dir)
    )
    
    return clustered_df, clustering_model

def create_visualizations(args, clustered_df=None, clustering_model=None):
    """Create visualizations of clustering results"""
    logger.info("Creating visualizations...")
    
    # Load clustered data if not provided
    if clustered_df is None:
        clustered_path = os.path.join(args.output_dir, 'clustering', 'clustered_data.csv')
        if os.path.exists(clustered_path):
            clustered_df = pd.read_csv(clustered_path)
            logger.info(f"Loaded clustered data from {clustered_path}")
        else:
            logger.error(f"Clustered data file not found at {clustered_path}")
            return
    
    # Create output directory for visualizations
    viz_output_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Check if UMAP coordinates exist
    if 'umap_x' not in clustered_df.columns or 'umap_y' not in clustered_df.columns:
        logger.error("UMAP coordinates not found in clustered DataFrame. Cannot create visualizations.")
        return
    
    logger.info("Creating UMAP visualizations...")
    
    # Convert 'bbox' to 'bbox_name' if necessary
    if 'bbox_name' not in clustered_df.columns and 'bbox' in clustered_df.columns:
        clustered_df['bbox_name'] = clustered_df['bbox']
    
    # Convert cluster to string type to ensure it's treated as categorical
    if 'cluster' in clustered_df.columns:
        clustered_df['cluster_category'] = 'Cluster ' + clustered_df['cluster'].astype(str)
    
    # Set color palette for distinct colors - using qualitative color maps
    # Get qualitative color maps
    distinct_cmap = cm.get_cmap('tab10')  # Good for up to 10 categories
    
    # Draw UMAP plot colored by bounding box
    if 'bbox_name' in clustered_df.columns:
        unique_bboxes = clustered_df['bbox_name'].unique()
        
        # Create matplotlib figure for bbox-colored visualization
        plt.figure(figsize=(12, 10))
        
        for i, bbox in enumerate(unique_bboxes):
            subset = clustered_df[clustered_df['bbox_name'] == bbox]
            color = distinct_cmap(i % 10)  # Cycle through 10 distinct colors
            plt.scatter(subset['umap_x'], subset['umap_y'], label=bbox, color=color, alpha=0.7)
        
        plt.title('UMAP 2D Projection - Features by Bounding Box', fontsize=16)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.legend(title='Bounding Box')
        plt.tight_layout()
        
        # Save the feature visualization
        feature_viz_path = os.path.join(viz_output_dir, 'umap_features_by_bbox.png')
        plt.savefig(feature_viz_path, dpi=300)
        plt.close()
        logger.info(f"Feature visualization saved to {feature_viz_path}")
        
        # Create interactive Plotly visualizations with discrete color scales
        logger.info("Creating interactive visualizations...")
        
        # Create bbox-colored visualization with DISCRETE color mapping
        fig = px.scatter(
            clustered_df, x='umap_x', y='umap_y', color='bbox_name',
            title='UMAP 2D Projection by Bounding Box',
            labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', 'bbox_name': 'Bounding Box'},
            color_discrete_sequence=px.colors.qualitative.Plotly  # Use Plotly's qualitative color sequence
        )
        
        # Save the bbox-colored visualization
        bbox_viz_path = os.path.join(viz_output_dir, 'umap_by_bbox.html')
        fig.write_html(bbox_viz_path)
        logger.info(f"UMAP by bbox visualization saved to {bbox_viz_path}")
    else:
        logger.warning("No 'bbox_name' column found. Skipping bbox visualization.")
    
    # Create the cluster-colored visualization if 'cluster' column exists
    if 'cluster' in clustered_df.columns:
        # Create cluster-colored visualization with DISCRETE colors
        fig2 = px.scatter(
            clustered_df, x='umap_x', y='umap_y', 
            # Use cluster_category instead of cluster for categorical colors
            color='cluster_category' if 'cluster_category' in clustered_df.columns else 'cluster',
            title='UMAP 2D Projection by Cluster',
            labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', 'cluster_category': 'Cluster'},
            color_discrete_sequence=px.colors.qualitative.Bold  # Use a different qualitative palette
        )
        
        # Save the cluster-colored visualization
        cluster_viz_path = os.path.join(viz_output_dir, 'umap_by_cluster.html')
        fig2.write_html(cluster_viz_path)
        logger.info(f"UMAP by cluster visualization saved to {cluster_viz_path}")
        
        # Static matplotlib comparison - if both bbox and cluster columns exist
        if 'bbox_name' in clustered_df.columns:
            plt.figure(figsize=(20, 10))
            
            # Left subplot: colored by bbox
            plt.subplot(1, 2, 1)
            for i, bbox in enumerate(unique_bboxes):
                subset = clustered_df[clustered_df['bbox_name'] == bbox]
                color = distinct_cmap(i % 10)
                plt.scatter(subset['umap_x'], subset['umap_y'], label=bbox, color=color, alpha=0.7)
            
            plt.title('UMAP by Bounding Box', fontsize=16)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.legend(title='Bounding Box')
            
            # Right subplot: colored by cluster with DISTINCT colors
            plt.subplot(1, 2, 2)
            
            if 'cluster_category' in clustered_df.columns:
                cluster_col = 'cluster_category'
            else:
                cluster_col = 'cluster'
                
            unique_clusters = clustered_df[cluster_col].unique()
            for i, cluster in enumerate(unique_clusters):
                subset = clustered_df[clustered_df[cluster_col] == cluster]
                color = distinct_cmap(i % 10)  # Cycle through distinct colors
                plt.scatter(subset['umap_x'], subset['umap_y'], label=str(cluster), color=color, alpha=0.7)
            
            plt.title('UMAP by Cluster', fontsize=16)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.legend(title='Cluster')
            
            plt.tight_layout()
            
            # Save the comparison visualization
            comparison_viz_path = os.path.join(viz_output_dir, 'umap_comparison.png')
            plt.savefig(comparison_viz_path, dpi=300)
            plt.close()
            logger.info(f"Comparison visualization saved to {comparison_viz_path}")
    else:
        logger.warning("No 'cluster' column found. Skipping cluster visualization.")
    
    # Create 3D visualizations if available
    if 'umap_z' in clustered_df.columns and args.create_3d_plots:
        logger.info("Creating 3D visualizations...")
        threed_dir = os.path.join(viz_output_dir, '3d')
        os.makedirs(threed_dir, exist_ok=True)
        
        # Choose the color column
        if 'cluster_category' in clustered_df.columns:
            color_column = 'cluster_category'
        elif 'cluster' in clustered_df.columns:
            color_column = 'cluster_category'  # We created this above
        else:
            color_column = 'bbox_name'
            
        # Create 3D plot with discrete color scheme
        fig_3d = px.scatter_3d(
            clustered_df, 
            x='umap_x', 
            y='umap_y', 
            z='umap_z', 
            color=color_column,
            title='3D UMAP Visualization',
            color_discrete_sequence=px.colors.qualitative.Vivid  # Another distinctive color palette
        )
        
        # Save 3D plot
        fig_3d_path = os.path.join(threed_dir, 'umap_3d.html')
        fig_3d.write_html(fig_3d_path)
        logger.info(f"3D visualization saved to {fig_3d_path}")
    
    logger.info(f"All visualizations saved to {viz_output_dir}")

def main():
    """Main function to run the complete analysis pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file logging
    log_file = setup_file_logger(args.output_dir)
    logger.info(f"Logging to {log_file}")
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Log the arguments
    logger.info("Starting SynapseClusterEM analysis with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Save the arguments for reproducibility
    args_file = os.path.join(args.output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run the selected mode(s)
    features_df = None
    clustered_df = None
    clustering_model = None
    
    try:
        if args.mode in ['preprocess', 'all']:
            args = preprocess_data(args)
        
        if args.mode in ['extract', 'all']:
            features_df = extract_features(args)
        
        if args.mode in ['cluster', 'all'] and (features_df is not None or args.mode != 'all'):
            clustered_df, clustering_model = perform_cluster_analysis(args, features_df)
        
        if args.mode in ['visualize', 'all'] and (clustered_df is not None or args.mode != 'all'):
            create_visualizations(args, clustered_df, clustering_model)
        
        logger.info("SynapseClusterEM analysis completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 