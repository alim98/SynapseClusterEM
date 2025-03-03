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

# Import project modules
from synapse_analysis.models.vgg3d import Vgg3D, load_model_from_checkpoint
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data
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
                # For other arguments that were explicitly provided
                elif not (isinstance(value, (bool, list)) and value == parser.get_default(key)):
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
        logger.info("Calculating global normalization statistics...")
        
        # Import the apply_global_normalization module
        sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
        from apply_global_normalization import calculate_global_stats
        
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
        global_stats_path = os.path.join(args.output_dir, 'global_stats.json')
        with open(global_stats_path, 'w') as f:
            json.dump(global_stats, f)
        
        args.global_stats_path = global_stats_path
        logger.info(f"Global statistics saved to {global_stats_path}")
    
    return args

def extract_features(args):
    """Extract features from synapse volumes using the VGG3D model"""
    logger.info("Starting feature extraction...")
    
    # Set device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    model = load_model_from_checkpoint(args.checkpoint_path, device)
    model.eval()
    
    # Load global normalization stats if needed
    global_stats = None
    if args.use_global_norm and args.global_stats_path:
        with open(args.global_stats_path, 'r') as f:
            global_stats = json.load(f)
        logger.info(f"Loaded global statistics from {args.global_stats_path}")
    
    # Extract features for each segmentation type
    features_df_list = []
    
    for seg_type in args.segmentation_types:
        logger.info(f"Processing segmentation type {seg_type}...")
        
        # Create output directory for this segmentation type
        seg_output_dir = os.path.join(args.output_dir, f"seg_type_{seg_type}")
        os.makedirs(seg_output_dir, exist_ok=True)
        
        # Load synapse data
        synapse_data = load_synapse_data(
            excel_dir=args.excel_dir,
            bbox_names=args.bbox_names
        )
        
        # Load volumes
        volumes = load_all_volumes(
            synapse_data=synapse_data,
            raw_base_dir=args.raw_base_dir,
            seg_base_dir=args.seg_base_dir,
            add_mask_base_dir=args.add_mask_base_dir,
            segmentation_type=seg_type
        )
        
        # Create dataset
        dataset = SynapseDataset(
            volumes=volumes,
            size=args.size,
            subvol_size=args.subvol_size,
            global_stats=global_stats
        )
        
        # Extract features
        features_df = extract_and_save_features(
            dataset=dataset,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=seg_output_dir
        )
        
        # Add segmentation type to features
        features_df['segmentation_type'] = seg_type
        features_df_list.append(features_df)
    
    # Combine features from all segmentation types
    if len(features_df_list) > 0:
        combined_features_df = pd.concat(features_df_list, ignore_index=True)
        combined_features_path = os.path.join(args.output_dir, 'combined_features.csv')
        combined_features_df.to_csv(combined_features_path, index=False)
        logger.info(f"Combined features saved to {combined_features_path}")
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
    
    # Perform clustering
    logger.info(f"Performing {args.clustering_method} clustering with {args.n_clusters} clusters...")
    clustered_df, clustering_model = perform_clustering(
        features_df,
        method=args.clustering_method,
        n_clusters=args.n_clusters,
        feature_cols=feature_cols
    )
    
    # Save clustered data
    clustered_path = os.path.join(cluster_output_dir, 'clustered_data.csv')
    clustered_df.to_csv(clustered_path, index=False)
    logger.info(f"Clustered data saved to {clustered_path}")
    
    # Compute embeddings for visualization
    logger.info("Computing embeddings for visualization...")
    embeddings_2d = compute_embeddings(
        features_df[feature_cols].values,
        n_components=2,
        method='umap'
    )
    
    embeddings_3d = None
    if args.create_3d_plots:
        embeddings_3d = compute_embeddings(
            features_df[feature_cols].values,
            n_components=3,
            method='umap'
        )
    
    # Add embeddings to DataFrame
    clustered_df['umap_x'] = embeddings_2d[:, 0]
    clustered_df['umap_y'] = embeddings_2d[:, 1]
    
    if embeddings_3d is not None:
        clustered_df['umap_z'] = embeddings_3d[:, 2]
    
    # Save updated clustered data with embeddings
    clustered_df.to_csv(clustered_path, index=False)
    
    # Analyze clusters
    logger.info("Analyzing clusters...")
    analyze_clusters(
        clustered_df,
        feature_cols,
        clustering_model,
        output_dir=cluster_output_dir
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
    
    # Save visualizations
    logger.info("Saving cluster visualizations...")
    save_cluster_visualizations(
        clustered_df,
        output_dir=viz_output_dir,
        create_3d=args.create_3d_plots,
        save_interactive=args.save_interactive
    )
    
    logger.info(f"Visualizations saved to {viz_output_dir}")

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