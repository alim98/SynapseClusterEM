"""
Main script for running the synapse analysis pipeline.

This script demonstrates how to use the SynapsePipeline class
to orchestrate the entire synapse analysis workflow.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from glob import glob
import time
import datetime  # Added for timestamp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
import seaborn as sns

from synapse import config
from synapse_pipeline import SynapsePipeline
from vesicle_size_visualizer import (
    compute_vesicle_cloud_sizes,
    create_umap_with_vesicle_sizes,
    analyze_vesicle_sizes_by_cluster,
    count_bboxes_in_clusters,
    plot_bboxes_in_clusters
)

# Import from newdl module
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from synapse import (
    Vgg3D, 
    load_model_from_checkpoint,
    config
)

# Import VGG3DStageExtractor for feature extraction
from vgg3d_stage_extractor import VGG3DStageExtractor

# Optional import for clustering utilities 
try:
    from synapse.clustering import (
        load_and_cluster_features,
        find_random_samples_in_clusters
    )
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    print("Warning: synapse.clustering module not found. Clustering functionality will be limited.")

# Set up logging to a file
log_file = open("pipeline_log.txt", "a")

def log_print(*args, **kwargs):
    """Custom print function that prints to both console and log file"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)
    log_file.flush()  # Make sure it's written immediately


def configure_pipeline_args():
    """Configure pipeline arguments by extending the existing config object"""
    # Create a new parser
    parser = argparse.ArgumentParser(description="Run the synapse analysis pipeline")
    
    # Add pipeline-specific arguments
    parser.add_argument("--only_vesicle_analysis", action="store_true",
                        help="Only run vesicle size analysis on existing results")
    
    # Add feature extraction method arguments
    parser.add_argument("--extraction_method", type=str, choices=['standard', 'stage_specific'],
                       help="Method to extract features ('standard' or 'stage_specific')")
    parser.add_argument("--layer_num", type=int,
                       help="Layer number to extract features from when using stage_specific method")
    
    # Let config parse its arguments first
    config.parse_args()
    
    # Parse our additional arguments
    args, _ = parser.parse_known_args()
    
    # Add our arguments to config
    if args.only_vesicle_analysis:
        config.only_vesicle_analysis = True
    else:
        config.only_vesicle_analysis = False
    
    # Add feature extraction parameters if provided
    if args.extraction_method:
        config.extraction_method = args.extraction_method
    
    if args.layer_num:
        config.layer_num = args.layer_num
    
    return config


def run_vesicle_analysis():
    """
    Run vesicle analysis on existing features.
    This is a stub function that will be implemented with the full vesicle analysis logic.
    """
    log_print("Vesicle analysis functionality will be implemented soon.")
    pass


def analyze_vesicle_sizes(pipeline, features_df):
    """
    Analyze vesicle sizes using the provided features.
    
    Args:
        pipeline: SynapsePipeline instance
        features_df: DataFrame with extracted features
        
    Returns:
        dict: Analysis results
    """
    log_print("Vesicle size analysis functionality will be implemented soon.")
    return {"status": "success"}


def setup_config():
    """
    Set up configuration for the pipeline
    """
    log_print("Setting up configuration...")
    
    # Parse config from command line args
    configure_pipeline_args()
    
    # Ensure the VGG3D model settings are correct
    config.extraction_method = "stage_specific"
    config.layer_num = 20
    
    # Create output directories
    os.makedirs(config.csv_output_dir, exist_ok=True)
    os.makedirs(config.clustering_output_dir, exist_ok=True)
    os.makedirs(config.save_gifs_dir, exist_ok=True)
    os.makedirs('manual/connection_visualizations', exist_ok=True)
    
    return config


def load_model():
    """
    Load the VGG3D model from checkpoint
    """
    log_print("Loading VGG3D model...")
    
    checkpoint_path = 'hemibrain_production.checkpoint'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)
    
    return model


def load_data():
    """
    Load the synapse data using the dataloader
    """
    log_print("Loading synapse data...")
    
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    vol_data_dict = {}
    for bbox_name in tqdm(config.bbox_name, desc="Loading volumes"):
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
    
    # Load synapse metadata from excel files
    syn_df = pd.concat([
        pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
        for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
    ])
    
    return vol_data_dict, syn_df


def load_manual_annotations():
    """
    Load manual annotations if available
    """
    log_print("Loading manual annotations...")
    
    manual_file = 'manual/clustering_results/manual_clustered_samples.csv'
    if not os.path.exists(manual_file):
        # Look in alternative locations
        alt_files = [
            'manual/manual_clustered_samples.csv',
            'manual/clustered_samples.csv'
        ]
        for alt_file in alt_files:
            if os.path.exists(alt_file):
                manual_file = alt_file
                break
        else:
            log_print("Warning: Manual annotation file not found. Continuing without manual annotations.")
            return None
    
    manual_df = pd.read_csv(manual_file)
    log_print(f"Loaded {len(manual_df)} manually annotated samples")
    return manual_df


def extract_stage_specific_features(model, vol_data_dict, syn_df, layer_num=20):
    """
    Extract features from a specific layer using the VGG3DStageExtractor
    """
    log_print(f"Extracting stage-specific features from layer {layer_num}...")
    
    # Create the stage extractor
    extractor = VGG3DStageExtractor(model)
    
    # Print information about the model stages
    stage_info = extractor.get_stage_info()
    for stage_num, info in stage_info.items():
        start_idx, end_idx = info['range']
        if start_idx <= layer_num <= end_idx:
            log_print(f"Layer {layer_num} is in Stage {stage_num}")
            break
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    
    # Create dataset with intelligent cropping parameters if specified
    dataset_kwargs = {
        'vol_data_dict': vol_data_dict,
        'synapse_df': syn_df,
        'processor': processor,
        'segmentation_type': config.segmentation_type,
        'alpha': config.alpha
    }
    
    # Apply intelligent cropping if specified in config
    if hasattr(config, 'preprocessing') and config.preprocessing == 'intelligent_cropping':
        dataset_kwargs.update({
            'smart_crop': True,
            'presynapse_weight': getattr(config, 'preprocessing_weights', 0.7),
            'normalize_presynapse_size': True
        })
    
    dataset = SynapseDataset(**dataset_kwargs)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Extract features for each synapse
    features = []
    metadata = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            # Get sample
            sample = dataset[i]
            if sample is None:
                continue
                
            pixels, info, name = sample
            
            # Add batch dimension and move to device
            inputs = pixels.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            
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
            metadata.append((name, info))
    
    if not features:
        raise ValueError("No features were extracted. Check your dataset and model.")
    
    # Combine features and metadata
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
    
    # Save features
    output_file = os.path.join(config.csv_output_dir, f"all_synapses_features_layer{layer_num}.csv")
    combined_df.to_csv(output_file, index=False)
    
    log_print(f"Extracted features for {len(combined_df)} synapses with {len(feature_columns)} features")
    log_print(f"Features saved to {output_file}")
    
    return combined_df


def project_features(features_df, method='umap'):
    """
    Project features to 2D space using dimensionality reduction
    """
    log_print(f"Projecting features using {method.upper()}...")
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    log_print(f"Using {len(feature_cols)} features for projection")
    
    # Standardize features
    features = features_df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features_scaled)
    else:  # Default to PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
    
    # Add projection coordinates to DataFrame
    features_df[f'{method}_1'] = embedding[:, 0]
    features_df[f'{method}_2'] = embedding[:, 1]
    
    # Save the projection
    output_file = os.path.join(config.csv_output_dir, f"all_synapses_projection_{method}.csv")
    features_df.to_csv(output_file, index=False)
    
    log_print(f"Projection saved to {output_file}")
    return features_df, reducer


def visualize_projection(features_df, manual_df, method='umap', output_dir=None):
    """
    Visualize the projection of all synapses, highlighting manual annotations
    
    Args:
        features_df: DataFrame with all synapse features and projections
        manual_df: DataFrame with manual annotations
        method: 'umap' or 'pca' for dimensionality reduction method
        output_dir: Directory to save visualizations (defaults to config.clustering_output_dir)
    """
    if output_dir is None:
        output_dir = config.clustering_output_dir
    
    log_print(f"Visualizing {method.upper()} projection with manual annotations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coordinate columns
    x_col = f'{method}_1'
    y_col = f'{method}_2'
    
    # Create a merged DataFrame with manual clusters if available
    if manual_df is not None:
        # Extract relevant columns from manual_df
        manual_info = manual_df[['bbox_name', 'Var1', 'Manual_Cluster']]
        
        # Merge with features_df
        merged_df = features_df.merge(
            manual_info, 
            on=['bbox_name', 'Var1'], 
            how='left'
        )
        
        # Identify manually annotated samples
        merged_df['is_manual'] = ~merged_df['Manual_Cluster'].isna()
    else:
        merged_df = features_df.copy()
        merged_df['is_manual'] = False
        merged_df['Manual_Cluster'] = np.nan
    
    # Get unique manual clusters and assign colors
    if manual_df is not None:
        clusters = sorted(manual_df['Manual_Cluster'].unique())
        cluster_colors = {cluster: plt.cm.tab10(i % 10) for i, cluster in enumerate(clusters)}
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Plot all synapses with small, light gray markers
    plt.scatter(
        merged_df[~merged_df['is_manual']][x_col],
        merged_df[~merged_df['is_manual']][y_col],
        color='lightgray',
        alpha=0.4,
        s=15,
        label='All Synapses'
    )
    
    # If manual annotations available, plot them and their connections
    if manual_df is not None and any(merged_df['is_manual']):
        # Get manually annotated samples
        manual_samples = merged_df[merged_df['is_manual']]
        
        # Plot connections between points in the same manual cluster
        for cluster in clusters:
            # Get points in this cluster
            cluster_points = manual_samples[manual_samples['Manual_Cluster'] == cluster]
            
            # Skip if only one point
            if len(cluster_points) <= 1:
                continue
                
            # Plot lines connecting all points in this cluster
            for i, row1 in cluster_points.iterrows():
                for j, row2 in cluster_points.iterrows():
                    if i < j:  # Only draw each connection once
                        plt.plot(
                            [row1[x_col], row2[x_col]],
                            [row1[y_col], row2[y_col]],
                            color=cluster_colors[cluster],
                            alpha=0.6,
                            linestyle='-',
                            linewidth=1.5
                        )
        
        # Plot manually annotated points with larger, colored markers
        for cluster, group in manual_samples.groupby('Manual_Cluster'):
            plt.scatter(
                group[x_col],
                group[y_col],
                color=cluster_colors[cluster],
                edgecolor='black',
                s=120,
                alpha=0.9,
                label=f"Manual Cluster {cluster}"
            )
            
            # Add synapse labels
            for i, row in group.iterrows():
                # Get a short version of the synapse name
                if "synapse" in row['Var1']:
                    parts = row['Var1'].split('_')
                    if len(parts) > 1:
                        label = parts[-1]  # Use the last part (often a number)
                    else:
                        label = row['Var1'][-5:]  # Use last 5 chars if no underscore
                else:
                    label = row['Var1'][-5:]  # Use last 5 chars
                
                # Add text with black outline for better visibility
                txt = plt.text(
                    row[x_col], row[y_col], 
                    label,
                    fontsize=9, 
                    ha='center', 
                    va='center',
                    fontweight='bold',
                    color='white'
                )
                
                # Add outline to text
                txt.set_path_effects([
                    PathEffects.withStroke(linewidth=2, foreground='black')
                ])
    
    # Add title and legend
    plt.title(f"All Synapses {method.upper()} Projection with Manual Cluster Connections", fontsize=16)
    plt.legend(loc='best', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"all_synapses_{method}_with_manual_clusters.png")
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.replace('.png', '.pdf'))
    
    log_print(f"Visualization saved to {output_file}")
    
    # Create a version with bbox information if manual annotations available
    if manual_df is not None and any(merged_df['is_manual']):
        create_detailed_visualization(merged_df, clusters, cluster_colors, x_col, y_col, method, output_dir)


def create_detailed_visualization(merged_df, clusters, cluster_colors, x_col, y_col, method, output_dir):
    """
    Create a more detailed visualization with bbox information
    """
    log_print("Creating detailed visualization with bbox information...")
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Use different marker styles for different bboxes
    bbox_markers = {
        'bbox1': 'o',  # circle
        'bbox2': 's',  # square
        'bbox3': '^',  # triangle up
        'bbox4': 'v',  # triangle down
        'bbox5': 'D',  # diamond
        'bbox6': 'p',  # pentagon
        'bbox7': 'h',  # hexagon
    }
    
    # Plot all synapses as small gray dots with markers by bbox
    for bbox, group in merged_df[~merged_df['is_manual']].groupby('bbox_name'):
        marker = bbox_markers.get(bbox, 'o')
        plt.scatter(
            group[x_col],
            group[y_col],
            color='lightgray',
            marker=marker,
            alpha=0.3,
            s=15
        )
    
    # Get manually annotated samples
    manual_samples = merged_df[merged_df['is_manual']]
    
    # Plot connections between points in the same manual cluster
    for cluster in clusters:
        cluster_points = manual_samples[manual_samples['Manual_Cluster'] == cluster]
        if len(cluster_points) <= 1:
            continue
            
        for i, row1 in cluster_points.iterrows():
            for j, row2 in cluster_points.iterrows():
                if i < j:
                    plt.plot(
                        [row1[x_col], row2[x_col]],
                        [row1[y_col], row2[y_col]],
                        color=cluster_colors[cluster],
                        alpha=0.6,
                        linestyle='-',
                        linewidth=1.5
                    )
    
    # Plot manually annotated points with markers by bbox and colors by cluster
    for (cluster, bbox), group in manual_samples.groupby(['Manual_Cluster', 'bbox_name']):
        marker = bbox_markers.get(bbox, 'o')
        plt.scatter(
            group[x_col],
            group[y_col],
            color=cluster_colors[cluster],
            marker=marker,
            edgecolor='black',
            s=120,
            alpha=0.9,
            label=f"Cluster {cluster} - {bbox}"
        )
        
        # Add labels
        for i, row in group.iterrows():
            if "synapse" in row['Var1']:
                parts = row['Var1'].split('_')
                if len(parts) > 1:
                    label = parts[-1]
                else:
                    label = row['Var1'][-5:]
            else:
                label = row['Var1'][-5:]
            
            txt = plt.text(
                row[x_col], row[y_col], 
                label,
                fontsize=9, 
                ha='center', 
                va='center',
                fontweight='bold',
                color='white'
            )
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='black')
            ])
    
    # Create a custom legend
    legend_elements = []
    
    # Add cluster color legend
    for cluster in clusters:
        legend_elements.append(
            Patch(facecolor=cluster_colors[cluster], edgecolor='black', label=f'Cluster {cluster}')
        )
    
    # Add bbox marker legend
    for bbox, marker in bbox_markers.items():
        if bbox in merged_df['bbox_name'].values:
            legend_elements.append(
                Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                      markeredgecolor='black', markersize=10, label=f'{bbox}')
            )
    
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    plt.title(f"All Synapses {method.upper()} Projection with BBox and Manual Cluster Information", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"all_synapses_{method}_with_bbox_and_manual_clusters.png")
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.replace('.png', '.pdf'))
    
    log_print(f"Detailed visualization saved to {output_file}")


def cluster_and_visualize(features_df, n_clusters=10, output_dir=None):
    """
    Perform clustering on the features and visualize the results
    """
    if output_dir is None:
        output_dir = config.clustering_output_dir
    
    log_print(f"Clustering features into {n_clusters} clusters...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    
    # Use the built-in clustering module if available
    if HAS_CLUSTERING:
        # First save the features to a temporary file for the clustering module
        temp_file = os.path.join(output_dir, "temp_features_for_clustering.csv")
        features_df.to_csv(temp_file, index=False)
        
        # Use the clustering module
        clustered_df, kmeans, _ = load_and_cluster_features(temp_file, n_clusters=n_clusters)
        
        # Save the clustered data
        clustered_file = os.path.join(output_dir, f"all_synapses_clustered_{n_clusters}.csv")
        clustered_df.to_csv(clustered_file, index=False)
        
        log_print(f"Clustering saved to {clustered_file}")
        return clustered_df
    else:
        # If clustering module not available, implement basic KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # Standardize features
        features = features_df[feature_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to DataFrame
        features_df['cluster'] = cluster_labels
        
        # Save clustered features
        clustered_file = os.path.join(output_dir, f"all_synapses_clustered_{n_clusters}.csv")
        features_df.to_csv(clustered_file, index=False)
        
        log_print(f"Clustering saved to {clustered_file}")
        return features_df


def visualize_manual_vs_auto_clusters(features_df, manual_df, method='umap', output_dir=None):
    """
    Visualize manually annotated clusters vs. automatic clusters
    """
    if output_dir is None:
        output_dir = config.clustering_output_dir
    
    if manual_df is None or 'cluster' not in features_df.columns:
        log_print("Skipping manual vs. auto cluster visualization (missing data)")
        return
    
    log_print("Visualizing manual vs. automatic clusters...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coordinate columns
    x_col = f'{method}_1'
    y_col = f'{method}_2'
    
    # Merge manual clusters with features
    merged_df = features_df.merge(
        manual_df[['bbox_name', 'Var1', 'Manual_Cluster']],
        on=['bbox_name', 'Var1'],
        how='left'
    )
    
    # Get manually annotated samples
    manual_samples = merged_df[~merged_df['Manual_Cluster'].isna()]
    
    if len(manual_samples) == 0:
        log_print("No overlap between manual annotations and automatic clusters")
        return
    
    # Create a confusion matrix of manual vs. automatic clusters
    conf_matrix = pd.crosstab(
        manual_samples['Manual_Cluster'],
        manual_samples['cluster'],
        rownames=['Manual'],
        colnames=['Auto']
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title("Confusion Matrix: Manual vs. Automatic Clusters")
    plt.tight_layout()
    
    # Save figure
    matrix_file = os.path.join(output_dir, f"manual_vs_auto_clusters_matrix.png")
    plt.savefig(matrix_file, dpi=300)
    
    # Create a visualization showing both clusterings
    plt.figure(figsize=(16, 14))
    
    # Create a custom colormap for manual clusters
    manual_clusters = sorted(manual_samples['Manual_Cluster'].unique())
    manual_cmap = plt.cm.get_cmap('Set1', len(manual_clusters))
    manual_colors = {cluster: manual_cmap(i) for i, cluster in enumerate(manual_clusters)}
    
    # Create a custom colormap for automatic clusters
    auto_clusters = sorted(manual_samples['cluster'].unique())
    auto_cmap = plt.cm.get_cmap('tab20', len(auto_clusters))
    auto_colors = {cluster: auto_cmap(i) for i, cluster in enumerate(auto_clusters)}
    
    # Draw lines connecting the same point in both visualizations
    # First create two side-by-side plots
    offset = np.max(manual_samples[x_col]) - np.min(manual_samples[x_col]) + 5
    
    # Left side: Manual clusters
    for cluster, group in manual_samples.groupby('Manual_Cluster'):
        plt.scatter(
            group[x_col],
            group[y_col],
            color=manual_colors[cluster],
            edgecolor='black',
            s=100,
            alpha=0.8,
            marker='o',
            label=f"Manual {cluster}"
        )
    
    # Right side: Automatic clusters
    for cluster, group in manual_samples.groupby('cluster'):
        plt.scatter(
            group[x_col] + offset,
            group[y_col],
            color=auto_colors[cluster],
            edgecolor='black',
            s=100,
            alpha=0.8,
            marker='s',
            label=f"Auto {cluster}"
        )
    
    # Connect the same points
    for i, row in manual_samples.iterrows():
        plt.plot(
            [row[x_col], row[x_col] + offset],
            [row[y_col], row[y_col]],
            color='gray',
            alpha=0.3,
            linestyle='-',
            linewidth=0.5
        )
    
    # Add labels
    plt.text(np.min(manual_samples[x_col]), np.max(manual_samples[y_col]) + 2, 
            "Manual Clusters", fontsize=16, ha='center')
    plt.text(np.min(manual_samples[x_col]) + offset, np.max(manual_samples[y_col]) + 2, 
            "Automatic Clusters", fontsize=16, ha='center')
    
    # Make two separate legends
    manual_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=manual_colors[cluster],
              markeredgecolor='black', markersize=10, label=f'Manual {cluster}')
        for cluster in manual_clusters
    ]
    
    auto_legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=auto_colors[cluster],
              markeredgecolor='black', markersize=10, label=f'Auto {cluster}')
        for cluster in auto_clusters
    ]
    
    # Place legends
    manual_legend = plt.legend(
        handles=manual_legend_elements, 
        loc='upper left', 
        fontsize=10, 
        title="Manual Clusters"
    )
    plt.gca().add_artist(manual_legend)
    
    auto_legend = plt.legend(
        handles=auto_legend_elements, 
        loc='upper right', 
        fontsize=10, 
        title="Auto Clusters"
    )
    
    plt.title("Comparison of Manual and Automatic Clusters")
    plt.grid(False)
    
    # Save figure
    comp_file = os.path.join(output_dir, f"manual_vs_auto_clusters_comparison.png")
    plt.savefig(comp_file, dpi=300)
    
    log_print(f"Cluster comparison saved to {comp_file}")


def run_pipeline():
    """
    Run the complete synapse analysis pipeline
    """
    log_print("Starting synapse analysis pipeline...")
    
    # Step 1: Setup
    setup_config()
    
    # Step 2: Load the VGG3D model
    model = load_model()
    
    # Step 3: Load synapse data
    vol_data_dict, syn_df = load_data()
    
    # Step 4: Load manual annotations (if available)
    manual_df = load_manual_annotations()
    
    # Step 5: Extract features
    features_df = extract_stage_specific_features(
        model, 
        vol_data_dict, 
        syn_df, 
        layer_num=config.layer_num
    )
    
    # Step 6: Project features to 2D
    features_df, _ = project_features(features_df, method='umap')
    
    # Step 7: Visualize projection with manual annotations
    visualize_projection(features_df, manual_df, method='umap')
    
    # Step 8: Cluster and visualize
    clustered_df = cluster_and_visualize(features_df, n_clusters=10)
    
    # Step 9: Compare manual and automatic clusters
    visualize_manual_vs_auto_clusters(clustered_df, manual_df, method='umap')
    
    log_print("Pipeline complete!")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        log_print(f"Error in pipeline: {str(e)}")
        import traceback
        log_print(traceback.format_exc())
    finally:
        log_file.close() 