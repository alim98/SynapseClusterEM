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
import shutil

# Import project modules
from synapse_analysis.models.vgg3d import Vgg3D, load_model_from_checkpoint
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data,
)
from synapse_analysis.data.dataset import SynapseDataset
from synapse_analysis.analysis.feature_extraction import extract_and_save_features
from synapse_analysis.analysis.clustering import (
    perform_clustering,
    compute_embeddings,
    analyze_clusters,
    save_cluster_visualizations
)

# Import the SynapseLogger class
from logger import SynapseLogger

# Import the ArgumentParser class
from argument_parser import ArgumentParser

# Initialize the logger
logger = SynapseLogger()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def extract_features(args):
    """Extract features from synapse volumes"""
    logger.info("Starting feature extraction...")
    logger.info(f"Using device: {args.device}")
    
    # Check GPU availability and set device
    if args.device.startswith('cuda'):
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.is_available()}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            device_name = torch.cuda.get_device_name(int(args.device.split(':')[1]) if ':' in args.device else 0)
            logger.info(f"GPU name: {device_name}")
            logger.info(f"gpu_id parameter value: {args.gpu_id}")
        else:
            logger.warning("GPU not available, falling back to CPU")
            args.device = 'cpu'
    
    # Load model
    model = Vgg3D(input_size=tuple(args.size) + (args.num_frames,), fmaps=24, 
                 output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, args.checkpoint_path)
    model.to(args.device)
    
    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = Path('outputs')
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if add_mask_base_dir exists
    if args.add_mask_base_dir and not os.path.exists(args.add_mask_base_dir):
        logger.warning(f"Additional mask directory {args.add_mask_base_dir} not found. Using None.")
        args.add_mask_base_dir = None
    
    feature_files = []
    for seg_type in args.segmentation_types:
        logger.info(f"Processing segmentation type {seg_type}...")
        seg_output_dir = output_dir / f"seg_type_{seg_type}"
        os.makedirs(seg_output_dir, exist_ok=True)
        
        # Download necessary files from Google Drive if in Colab
        if 'google.colab' in sys.modules:
            import google.colab
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            drive_dir = Path('/content/drive/MyDrive/SynapseClusterEM_outputs')
            os.makedirs(drive_dir, exist_ok=True)
        else:
            drive_dir = None
        
        # Create samples for visualization
        try:
            sample_vis_dir = seg_output_dir / "sample_visualizations"
            os.makedirs(sample_vis_dir, exist_ok=True)
            
            # Load first few samples to generate visualizations
            excel_dir = args.excel_dir
            print(f"Excel directory: {excel_dir}")
            print(f"Excel directory exists: {os.path.exists(excel_dir)}")
            
            for bbox in args.bbox_names[:2]:  # First 2 bounding boxes
                excel_path = os.path.join(excel_dir, f"{bbox}.xlsx")
                print(f"Looking for Excel file: {excel_path}")
                print(f"Excel file exists: {os.path.exists(excel_path)}")
                
            # Load synapse data
            synapse_data = load_synapse_data(args.bbox_names, args.excel_dir)
            if len(synapse_data) == 0:
                logger.error("No synapse data found. Check excel_dir path.")
                return []
                
            # Load volumes
            volumes = load_all_volumes(
                args.bbox_names, 
                args.raw_base_dir, 
                args.seg_base_dir, 
                args.add_mask_base_dir
            )
            if len(volumes) == 0:
                logger.error("No volumes loaded. Check data paths.")
                return []
            
            # Create visualizations for a few samples
            num_samples = min(20, len(synapse_data))
            sample_indices = np.random.choice(len(synapse_data), num_samples, replace=False)
            
            for i, idx in enumerate(sample_indices):
                syn_info = synapse_data.iloc[idx]
                bbox_name = syn_info['bbox_name']
                
                if bbox_name not in volumes:
                    continue
                    
                raw_vol, seg_vol, add_mask_vol = volumes[bbox_name]
                
                central_coord = (int(syn_info['central_coord_1']), 
                               int(syn_info['central_coord_2']), 
                               int(syn_info['central_coord_3']))
                side1_coord = (int(syn_info['side_1_coord_1']), 
                              int(syn_info['side_1_coord_2']), 
                              int(syn_info['side_1_coord_3']))
                side2_coord = (int(syn_info['side_2_coord_1']), 
                              int(syn_info['side_2_coord_2']), 
                              int(syn_info['side_2_coord_3']))
                
                # Create segmented cube
                from synapse_analysis.utils.processing import create_segmented_cube
                
                segmented_cube = create_segmented_cube(
                    raw_vol=raw_vol,
                    seg_vol=seg_vol,
                    add_mask_vol=add_mask_vol,
                    central_coord=central_coord,
                    side1_coord=side1_coord,
                    side2_coord=side2_coord,
                    segmentation_type=seg_type,
                    subvolume_size=args.subvol_size,
                    alpha=args.alphas[0],
                    bbox_name=bbox_name
                )
                
                # Save central slice
                central_slice = segmented_cube[:, :, segmented_cube.shape[2] // 2, :]
                slice_img = central_slice[:, :, central_slice.shape[2] // 2]
                
                # Log slice image statistics
                print(f"[VIZ DEBUG] Slice image stats for sample {i+1}, {bbox_name}, Synapse {syn_info['Var1']}:")
                print(f"[VIZ DEBUG] - Shape: {slice_img.shape}")
                print(f"[VIZ DEBUG] - Min value: {np.min(slice_img)}")
                print(f"[VIZ DEBUG] - Max value: {np.max(slice_img)}")
                print(f"[VIZ DEBUG] - Mean value: {np.mean(slice_img)}")
                print(f"[VIZ DEBUG] - Std dev: {np.std(slice_img)}")
                
                # Create a mask for the gray regions (where value is exactly 128.0)
                gray_mask = np.isclose(slice_img, 128.0)
                print(f"[VIZ DEBUG] - Gray mask coverage: {np.sum(gray_mask) / gray_mask.size * 100:.2f}%")
                
                # Normalize the slice to ensure consistent visualization
                # This ensures the gray value appears the same across all images
                normalized_slice = slice_img.copy()
                
                # Replace the gray value with a special value temporarily
                special_value = -1000  # A value that won't occur naturally
                normalized_slice[gray_mask] = special_value
                
                # Normalize the non-gray parts
                non_gray_mask = ~gray_mask
                if np.any(non_gray_mask):
                    min_val = np.min(normalized_slice[non_gray_mask])
                    max_val = np.max(normalized_slice[non_gray_mask])
                    
                    # Avoid division by zero
                    if max_val > min_val:
                        # Scale the non-gray parts to 0-255
                        normalized_slice[non_gray_mask] = 255.0 * (normalized_slice[non_gray_mask] - min_val) / (max_val - min_val)
                
                # Set the gray parts to a consistent value (128)
                normalized_slice[gray_mask] = 128.0
                
                print(f"[VIZ DEBUG] Normalized slice stats:")
                print(f"[VIZ DEBUG] - Min value: {np.min(normalized_slice)}")
                print(f"[VIZ DEBUG] - Max value: {np.max(normalized_slice)}")
                
                plt.figure(figsize=(8, 8))
                # Use the normalized slice with consistent gray value
                plt.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
                plt.title(f"Sample {i+1}: {bbox_name}, Synapse {syn_info['Var1']}")
                plt.axis('off')
                
                sample_vis_path = sample_vis_dir / f"sample_{i+1}_{bbox_name}_synapse_{syn_info['Var1']}.png"
                plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            logger.info(f"Sample visualizations saved to {sample_vis_dir}")
            # Copy visualizations to Google Drive if in Colab
            if drive_dir:
                drive_vis_dir = drive_dir / "sample_visualizations"
                os.makedirs(drive_vis_dir, exist_ok=True)
                for f in sample_vis_dir.glob("*.png"):
                    shutil.copy(f, drive_vis_dir / f.name)
            
            logger.info(f"Sample visualizations created at {drive_dir / 'sample_visualizations' if drive_dir else None}")
        except Exception as e:
            logger.warning(f"Error creating sample visualizations: {e}")
        
        # For each alpha value
        for alpha in args.alphas:
            try:
                # Check if features already exist
                feature_filename = f"features_segtype_{seg_type}_alpha_{alpha:.1f}.csv"
                feature_path = seg_output_dir / feature_filename
                
                if os.path.exists(feature_path) and not args.force_recompute:
                    logger.info(f"Features already exist at {feature_path}. Use --force_recompute to recompute.")
                    feature_files.append(str(feature_path))
                    continue
                    
                # Create dataset and extract features
                if 'synapse_data' not in locals() or 'volumes' not in locals():
                    # Load synapse data
                    synapse_data = load_synapse_data(
                        bbox_names=args.bbox_names,
                        excel_dir=args.excel_dir
                    )
                    
                    # Load volumes
                    volumes = load_all_volumes(
                        bbox_names=args.bbox_names,
                        raw_base_dir=args.raw_base_dir,
                        seg_base_dir=args.seg_base_dir,
                        add_mask_base_dir=args.add_mask_base_dir
                    )
                
                # Create a processor with sample-wise normalization disabled
                processor = Synapse3DProcessor(
                    size=tuple(args.size),
                    apply_sample_norm=False  # Disable sample-wise normalization
                )
                
                # Create dataset
                dataset = SynapseDataset(
                    vol_data_dict=volumes,
                    synapse_df=synapse_data,
                    processor=processor,
                    segmentation_type=seg_type,
                    subvol_size=args.subvol_size,
                    num_frames=args.num_frames,
                    alpha=alpha
                )
                
                # Extract features
                feature_path = extract_and_save_features(
                    model=model,
                    dataset=dataset,
                    seg_type=seg_type,
                    alpha=alpha,
                    output_dir=str(seg_output_dir),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    drive_dir=str(drive_dir) if drive_dir else None
                )
                
                feature_files.append(feature_path)
                
            except Exception as e:
                logger.error(f"Error processing segmentation type {seg_type} with alpha {alpha}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Combine features if we have multiple feature files
    if len(feature_files) > 0:
        try:
            combined_features_path = output_dir / "combined_features.csv"
            
            # Load and combine all feature files
            dfs = []
            for file_path in feature_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # Add segmentation type and alpha if they're not already in the file
                    if 'seg_type' not in df.columns:
                        seg_type_str = os.path.basename(file_path).split('_')[2]
                        df['seg_type'] = int(seg_type_str) if seg_type_str.isdigit() else seg_type_str
                    if 'alpha' not in df.columns:
                        alpha_str = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
                        df['alpha'] = float(alpha_str)
                    dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(combined_features_path, index=False)
                logger.info(f"Combined features saved to {combined_features_path}")
                
                # Copy to Google Drive if in Colab
                if drive_dir:
                    drive_combined_path = drive_dir / "combined_features.csv"
                    combined_df.to_csv(drive_combined_path, index=False)
            else:
                logger.warning("No feature files could be loaded. Combined features not created.")
        except Exception as e:
            logger.error(f"Error combining features: {e}")
    
    return feature_files

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
    """Main entry point of the application."""
    # Parse arguments using the ArgumentParser class
    arg_parser = ArgumentParser(logger)
    args = arg_parser.parse_args()
    
    # Set up logging to file if output directory is specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        log_file = logger.setup_file_logger(output_dir)
    
    # Fix any path issues in the configuration
    if hasattr(args, 'add_mask_base_dir') and args.add_mask_base_dir == "":
        args.add_mask_base_dir = None
        
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Log the parameters
    logger.info("Starting SynapseClusterEM analysis with the following parameters:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Backward compatibility for flattened config files
    try:
        if os.path.exists(args.config):
            config = arg_parser.load_config(args.config)
            if config and 'data_paths' in config:
                paths = config['data_paths']
                for key in ['raw_base_dir', 'seg_base_dir', 'add_mask_base_dir', 'excel_dir', 'output_dir', 'checkpoint_path']:
                    if key in paths:
                        setattr(args, key, paths[key])
    except (TypeError, AttributeError):
        logger.debug("Skipping backward compatibility check for config file.")
    
    # Run the pipeline according to the selected mode
    try:
        if args.mode in ['preprocess', 'all']:
            logger.info("Starting data preprocessing...")
            
        if args.mode in ['extract', 'all']:
            logger.info("Starting feature extraction...")
            feature_files = extract_features(args)
            
            # If feature extraction succeeded, get the DataFrame
            if feature_files:
                # Load combined features if they exist
                combined_file = os.path.join(args.output_dir, "combined_features.csv")
                if os.path.exists(combined_file):
                    features_df = pd.read_csv(combined_file)
                else:
                    # Try to load the first feature file
                    if os.path.exists(feature_files[0]):
                        features_df = pd.read_csv(feature_files[0])
                    else:
                        features_df = None
            else:
                features_df = None
                
        if args.mode in ['cluster', 'all'] and (args.mode != 'all' or 'features_df' in locals()):
            logger.info("Starting clustering analysis...")
            if args.mode == 'cluster' and 'features_df' not in locals():
                # Load combined features if we're only doing clustering
                combined_file = os.path.join(args.output_dir, "combined_features.csv")
                if os.path.exists(combined_file):
                    features_df = pd.read_csv(combined_file)
                else:
                    logger.error("No features found. Please run the 'extract' mode first.")
                    return
            
            clustered_df, clustering_model = perform_cluster_analysis(args, features_df)
            
        if args.mode in ['visualize', 'all'] and (args.mode != 'all' or 'clustered_df' in locals()):
            logger.info("Creating visualizations...")
            if args.mode == 'visualize' and 'clustered_df' not in locals():
                # Load clustered data if we're only doing visualization
                clustered_file = os.path.join(args.output_dir, "clustering", "clustered_data.csv")
                if os.path.exists(clustered_file):
                    clustered_df = pd.read_csv(clustered_file)
                else:
                    logger.error("No clustered data found. Please run the 'cluster' mode first.")
                    return
                    
                # Load features for visualization
                combined_file = os.path.join(args.output_dir, "combined_features.csv")
                if os.path.exists(combined_file):
                    features_df = pd.read_csv(combined_file)
                else:
                    logger.error("No features found. Visualizations will be limited.")
                    features_df = None
            
            create_visualizations(args, clustered_df, features_df)
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("SynapseClusterEM analysis completed.")

if __name__ == "__main__":
    sys.exit(main()) 