"""
Synapse Feature Projection Tool

This module provides functions to generate UMAP and t-SNE projections from synapse feature data.
It creates 2D and 3D visualizations colored by bounding box number and cluster number.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from datetime import datetime


def load_feature_data(csv_path):
    """
    Load feature data from CSV file and identify feature columns.
    
    Args:
        csv_path (str): Path to the CSV file containing feature data
        
    Returns:
        tuple: (DataFrame with all data, feature columns only, boolean indicating if cluster column exists)
    """
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Identify feature columns (assuming they start with 'feat_')
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    if not feature_cols:
        raise ValueError("No feature columns found in CSV. Feature columns should start with 'feat_'")
    
    # Check if required columns exist
    has_cluster = 'cluster' in df.columns
    has_bbox = 'bbox_name' in df.columns or 'bbox_number' in df.columns
    
    # Standardize bbox column name
    if 'bbox_number' in df.columns and not 'bbox_name' in df.columns:
        df['bbox_name'] = df['bbox_number']
    elif 'bbox_name' in df.columns and not 'bbox_number' in df.columns:
        df['bbox_number'] = df['bbox_name']
    
    if not has_bbox:
        raise ValueError("No bbox_name or bbox_number column found in CSV")
    
    # Check for vesicle_cloud_size
    has_vesicle_size = 'vesicle_cloud_size' in df.columns
    
    print(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    if has_cluster:
        print(f"Found {df['cluster'].nunique()} clusters")
    print(f"Found {df['bbox_name'].nunique()} unique bounding boxes")
    
    return df, feature_cols, has_cluster, has_vesicle_size


def apply_umap(features, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Apply UMAP dimensionality reduction to features.
    
    Args:
        features (numpy.ndarray): Feature matrix
        n_components (int): Number of dimensions for projection (2 or 3)
        n_neighbors (int): Number of neighbors for UMAP (automatically adjusted if too large)
        min_dist (float): Minimum distance parameter for UMAP
        random_state (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: UMAP projection
    """
    print(f"Applying UMAP ({n_components}D)...")
    
    # Ensure n_neighbors is less than the number of samples
    n_samples = features.shape[0]
    if n_neighbors >= n_samples:
        n_neighbors = max(1, n_samples - 1)
        print(f"Warning: n_neighbors reduced to {n_neighbors} (must be less than n_samples={n_samples})")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    projection = reducer.fit_transform(features)
    return projection


def apply_tsne(features, n_components=2, perplexity=30, random_state=42):
    """
    Apply t-SNE dimensionality reduction to features.
    
    Args:
        features (numpy.ndarray): Feature matrix
        n_components (int): Number of dimensions for projection (2 or 3)
        perplexity (int): Perplexity parameter for t-SNE
        random_state (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: t-SNE projection
    """
    print(f"Applying t-SNE ({n_components}D)...")
    
    # Adjust perplexity if it's larger than the number of samples
    n_samples = features.shape[0]
    if n_samples <= perplexity:
        # Set perplexity to n_samples / 3 (common rule of thumb)
        # but make sure it's at least 1
        adjusted_perplexity = max(1, n_samples // 3)
        print(f"Warning: Perplexity ({perplexity}) must be less than n_samples ({n_samples}). "
              f"Adjusting perplexity to {adjusted_perplexity}.")
        perplexity = adjusted_perplexity
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    projection = tsne.fit_transform(features)
    return projection


def plot_2d_projection(projection, color_values, size_values=None, title="", cmap="tab10", 
                       color_label="", output_path=None):
    """
    Plot a 2D projection with colors based on a categorical variable.
    
    Args:
        projection (numpy.ndarray): 2D projection
        color_values (numpy.ndarray): Values to color points by
        size_values (numpy.ndarray, optional): Values to size points by
        title (str): Plot title
        cmap (str): Colormap name
        color_label (str): Label for the color legend
        output_path (str, optional): Path to save the plot
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Create main plot area
    ax_main = plt.subplot2grid((8, 8), (0, 0), colspan=8, rowspan=7)
    
    # Handle case where all points have the same color value
    unique_colors = np.unique(color_values)
    if len(unique_colors) == 1:
        print(f"Warning: All points have the same {color_label} value ({unique_colors[0]})")
        # Use a single color (the middle of the colormap)
        try:
            single_color = plt.colormaps[cmap](0.5)
        except (AttributeError, KeyError):
            single_color = plt.cm.get_cmap(cmap)(0.5)
        
        if size_values is not None:
            # Normalize size between 10 and 100
            size = 10 + 90 * (size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-10)
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], color=single_color, alpha=0.7, s=size)
        else:
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], color=single_color, alpha=0.7)
        
        # Create a palette area - simple for single value
        ax_palette = plt.subplot2grid((8, 8), (7, 0), colspan=8, rowspan=1)
        ax_palette.add_patch(plt.Rectangle((0, 0), 0.8, 0.8, facecolor=single_color, edgecolor='black'))
        ax_palette.text(0.4, -0.5, str(unique_colors[0]), ha='center', va='top')
        ax_palette.set_xlim(-0.5, 1.5)
        ax_palette.set_ylim(-1, 1)
        ax_palette.axis('off')
        ax_palette.set_title(f"Color: {color_label}", pad=10)
        
    else:
        # Normal case with multiple colors
        if size_values is not None:
            # Normalize size between 10 and 100
            size = 10 + 90 * (size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-10)
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], c=color_values, cmap=cmap, 
                                alpha=0.7, s=size)
        else:
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], c=color_values, cmap=cmap, alpha=0.7)
        
        # Add a regular colorbar
        cbar = plt.colorbar(scatter, ax=ax_main, label=color_label)
        
        # Also create a palette with actual values
        ax_palette = plt.subplot2grid((8, 8), (7, 0), colspan=8, rowspan=1)
        
        # Calculate patch positions
        n_unique = len(unique_colors)
        patch_width = 6 / n_unique  # Use 6/8 of width for patches
        start_x = 1  # Start 1/8 from left
        
        # Create color patches with values
        for i, value in enumerate(unique_colors):
            # Position for current patch
            x_pos = start_x + i * patch_width
            
            # Get color from colormap
            norm_i = i / max(1, n_unique - 1)  # Avoid division by zero
            try:
                color = plt.colormaps[cmap](norm_i)
            except (AttributeError, KeyError):
                color = plt.cm.get_cmap(cmap)(norm_i)
                
            # Add the patch and text
            ax_palette.add_patch(plt.Rectangle((x_pos, 0), patch_width*0.8, 0.8, 
                                             facecolor=color, edgecolor='black'))
            ax_palette.text(x_pos + patch_width*0.4, -0.5, str(value), 
                          ha='center', va='top', fontsize=8)
        
        # Set up the palette axis
        ax_palette.set_xlim(0, 8)
        ax_palette.set_ylim(-1, 1)
        ax_palette.axis('off')
        ax_palette.set_title(f"Color palette: {color_label}", pad=10)
    
    # Configure main plot
    ax_main.set_title(title)
    ax_main.set_xlabel("Dimension 1")
    ax_main.set_ylabel("Dimension 2")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_3d_projection(projection, color_values, size_values=None, title="", cmap="tab10", 
                       color_label="", output_path=None):
    """
    Plot a 3D projection with colors based on a categorical variable.
    
    Args:
        projection (numpy.ndarray): 3D projection
        color_values (numpy.ndarray): Values to color points by
        size_values (numpy.ndarray, optional): Values to size points by
        title (str): Plot title
        cmap (str): Colormap name
        color_label (str): Label for the color legend
        output_path (str, optional): Path to save the plot
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Create main 3D plot
    ax_main = plt.subplot2grid((8, 8), (0, 0), colspan=8, rowspan=7, projection='3d')
    
    # Handle case where all points have the same color value
    unique_colors = np.unique(color_values)
    if len(unique_colors) == 1:
        print(f"Warning: All points have the same {color_label} value ({unique_colors[0]})")
        # Use a single color (the middle of the colormap)
        try:
            single_color = plt.colormaps[cmap](0.5)
        except (AttributeError, KeyError):
            single_color = plt.cm.get_cmap(cmap)(0.5)
        
        if size_values is not None:
            # Normalize size between 10 and 100
            size = 10 + 90 * (size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-10)
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], projection[:, 2], 
                                color=single_color, alpha=0.7, s=size)
        else:
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], projection[:, 2], 
                                color=single_color, alpha=0.7)
        
        # Create a palette area - simple for single value
        ax_palette = plt.subplot2grid((8, 8), (7, 0), colspan=8, rowspan=1)
        ax_palette.add_patch(plt.Rectangle((0, 0), 0.8, 0.8, facecolor=single_color, edgecolor='black'))
        ax_palette.text(0.4, -0.5, str(unique_colors[0]), ha='center', va='top')
        ax_palette.set_xlim(-0.5, 1.5)
        ax_palette.set_ylim(-1, 1)
        ax_palette.axis('off')
        ax_palette.set_title(f"Color: {color_label}", pad=10)
        
    else:
        # Normal case with multiple colors
        if size_values is not None:
            # Normalize size between 10 and 100
            size = 10 + 90 * (size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-10)
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], projection[:, 2], 
                                c=color_values, cmap=cmap, alpha=0.7, s=size)
        else:
            scatter = ax_main.scatter(projection[:, 0], projection[:, 1], projection[:, 2], 
                                c=color_values, cmap=cmap, alpha=0.7)
        
        # Add a regular colorbar
        cbar = plt.colorbar(scatter, ax=ax_main, label=color_label, pad=0.1)
        
        # Also create a palette with actual values
        ax_palette = plt.subplot2grid((8, 8), (7, 0), colspan=8, rowspan=1)
        
        # Calculate patch positions
        n_unique = len(unique_colors)
        patch_width = 6 / n_unique  # Use 6/8 of width for patches
        start_x = 1  # Start 1/8 from left
        
        # Create color patches with values
        for i, value in enumerate(unique_colors):
            # Position for current patch
            x_pos = start_x + i * patch_width
            
            # Get color from colormap
            norm_i = i / max(1, n_unique - 1)  # Avoid division by zero
            try:
                color = plt.colormaps[cmap](norm_i)
            except (AttributeError, KeyError):
                color = plt.cm.get_cmap(cmap)(norm_i)
                
            # Add the patch and text
            ax_palette.add_patch(plt.Rectangle((x_pos, 0), patch_width*0.8, 0.8, 
                                             facecolor=color, edgecolor='black'))
            ax_palette.text(x_pos + patch_width*0.4, -0.5, str(value), 
                          ha='center', va='top', fontsize=8)
        
        # Set up the palette axis
        ax_palette.set_xlim(0, 8)
        ax_palette.set_ylim(-1, 1)
        ax_palette.axis('off')
        ax_palette.set_title(f"Color palette: {color_label}", pad=10)
    
    # Configure main plot
    ax_main.set_title(title)
    ax_main.set_xlabel("Dimension 1")
    ax_main.set_ylabel("Dimension 2")
    ax_main.set_zlabel("Dimension 3")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def create_color_palette_legend(color_values, cmap, title, output_path=None):
    """
    Create a color palette legend showing the mapping between values and colors.
    
    Args:
        color_values (numpy.ndarray): Values used for coloring
        cmap (str): Colormap name
        title (str): Title for the legend
        output_path (str, optional): Path to save the legend plot
    """
    unique_values = np.unique(color_values)
    n_unique = len(unique_values)
    
    # Handle case with only one unique value
    if n_unique <= 1:
        fig, ax = plt.subplots(figsize=(3, 1))
        
        # Use the middle color of the colormap for a single value
        try:
            # Using the new recommended way to get colormaps
            color = plt.colormaps[cmap](0.5)
        except (AttributeError, KeyError):
            # Fallback for older matplotlib versions
            color = plt.cm.get_cmap(cmap)(0.5)
            
        ax.add_patch(plt.Rectangle((0, 0), 0.8, 0.8, facecolor=color, edgecolor='black'))
        if n_unique == 1:
            ax.text(0.4, -0.5, str(unique_values[0]), ha='center', va='top')
        else:
            ax.text(0.4, -0.5, "No values", ha='center', va='top')
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        ax.set_title(title, pad=20)
    else:
        # Create a figure with a single row of color patches
        fig, ax = plt.subplots(figsize=(n_unique * 1.5, 1))
        
        # Create color patches
        for i, value in enumerate(unique_values):
            # Safe division with n_unique > 1
            norm_i = i / (n_unique - 1)
            
            try:
                # Using the new recommended way to get colormaps
                color = plt.colormaps[cmap](norm_i)
            except (AttributeError, KeyError):
                # Fallback for older matplotlib versions
                color = plt.cm.get_cmap(cmap)(norm_i)
                
            ax.add_patch(plt.Rectangle((i, 0), 0.8, 0.8, facecolor=color, edgecolor='black'))
            ax.text(i + 0.4, -0.5, str(value), ha='center', va='top')
        
        # Set up the plot
        ax.set_xlim(-0.5, n_unique)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        ax.set_title(title, pad=20)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved color palette to {output_path}")
    else:
        plt.show()
    plt.close()


def create_projections(df, feature_cols, output_dir, has_cluster=True, has_vesicle_size=False,
                   perplexity=30, n_neighbors=15):
    """
    Create and save all projections (UMAP and t-SNE, 2D and 3D).
    
    Args:
        df (pandas.DataFrame): DataFrame containing feature data
        feature_cols (list): List of feature column names
        output_dir (str): Directory to save output plots
        has_cluster (bool): Whether cluster information is available
        has_vesicle_size (bool): Whether vesicle cloud size information is available
        perplexity (int): Perplexity parameter for t-SNE
        n_neighbors (int): Number of neighbors for UMAP
    """
    # Extract features for projection
    features = df[feature_cols].values
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Adjust UMAP n_neighbors if needed - should be less than n_samples
    n_samples = features.shape[0]
    umap_n_neighbors = min(n_neighbors, n_samples - 1)  # Ensure it's less than n_samples
    
    # Generate projections
    umap_2d = apply_umap(features, n_components=2, n_neighbors=umap_n_neighbors)
    umap_3d = apply_umap(features, n_components=3, n_neighbors=umap_n_neighbors)
    tsne_2d = apply_tsne(features, n_components=2, perplexity=perplexity)
    tsne_3d = apply_tsne(features, n_components=3, perplexity=perplexity)
    
    # Create plots for each projection, colored by bbox
    bbox_values = df['bbox_name'].values
    
    # Convert string bbox values to numeric IDs for coloring
    if bbox_values.dtype == object:  # If values are strings or mixed types
        unique_bboxes = np.unique(bbox_values)
        bbox_to_id = {bbox: i for i, bbox in enumerate(unique_bboxes)}
        bbox_numeric = np.array([bbox_to_id[bbox] for bbox in bbox_values])
        print(f"Converted {len(unique_bboxes)} unique bbox names to numeric IDs for plotting")
    else:
        bbox_numeric = bbox_values
    
    # Function to create all variations of a plot
    def create_all_variations(projection_2d, projection_3d, method_name, color_values, 
                              color_label, color_cmap="tab10"):
        # Standard plots (no size variation)
        plot_2d_projection(
            projection_2d, color_values, 
            title=f"{method_name} 2D Projection (colored by {color_label})",
            cmap=color_cmap, color_label=color_label,
            output_path=os.path.join(output_dir, f"{method_name}_2d_{color_label}.png")
        )
        
        plot_3d_projection(
            projection_3d, color_values,
            title=f"{method_name} 3D Projection (colored by {color_label})",
            cmap=color_cmap, color_label=color_label,
            output_path=os.path.join(output_dir, f"{method_name}_3d_{color_label}.png")
        )
        
        # Plots with vesicle size if available
        if has_vesicle_size:
            size_values = df['vesicle_cloud_size'].values
            
            plot_2d_projection(
                projection_2d, color_values, size_values,
                title=f"{method_name} 2D Projection (colored by {color_label}, sized by vesicle cloud)",
                cmap=color_cmap, color_label=color_label,
                output_path=os.path.join(output_dir, f"{method_name}_2d_{color_label}_sized.png")
            )
            
            plot_3d_projection(
                projection_3d, color_values, size_values,
                title=f"{method_name} 3D Projection (colored by {color_label}, sized by vesicle cloud)",
                cmap=color_cmap, color_label=color_label,
                output_path=os.path.join(output_dir, f"{method_name}_3d_{color_label}_sized.png")
            )
    
    # Create bbox-colored plots
    create_all_variations(umap_2d, umap_3d, "UMAP", bbox_numeric, "bbox", "tab20")
    create_all_variations(tsne_2d, tsne_3d, "TSNE", bbox_numeric, "bbox", "tab20")
    
    # Create cluster-colored plots if clusters are available
    if has_cluster:
        cluster_values = df['cluster'].values
        create_all_variations(umap_2d, umap_3d, "UMAP", cluster_values, "cluster")
        create_all_variations(tsne_2d, tsne_3d, "TSNE", cluster_values, "cluster")


def main():
    """Main function to parse arguments and run the projection tool"""
    parser = argparse.ArgumentParser(description="Generate UMAP and t-SNE projections from feature data")
    parser.add_argument("csv_file", help="Path to CSV file containing feature data")
    parser.add_argument("--output-dir", "-o", default=None, 
                        help="Directory to save output plots (default: 'projections_TIMESTAMP')")
    parser.add_argument("--perplexity", "-p", type=int, default=30,
                        help="Perplexity parameter for t-SNE (default: 30, will be automatically reduced for small datasets)")
    parser.add_argument("--n-neighbors", "-n", type=int, default=15,
                        help="Number of neighbors for UMAP (default: 15, will be automatically reduced for small datasets)")
    
    args = parser.parse_args()
    # Create output directory if not specified
    if args.output_dir is None:
        try:
            from synapse.utils.config import config
            if hasattr(config, 'output_dir'):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                args.output_dir = os.path.join(config.output_dir, f"projections_{timestamp}")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                args.output_dir = f"projections_{timestamp}"
        except ImportError:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.output_dir = f"projections_{timestamp}"
    # Load data and create projections
    df, feature_cols, has_cluster, has_vesicle_size = load_feature_data(args.csv_file)
    
    # Small dataset warning
    if len(df) < 20:
        print("WARNING: Small dataset detected (less than 20 samples).")
        print("  - t-SNE and UMAP may not produce meaningful results with very few samples")
        print("  - Parameters will be automatically adjusted to work with your small dataset")
        print("  - Consider collecting more data for more reliable projections")
    
    create_projections(
        df, feature_cols, args.output_dir, 
        has_cluster=has_cluster,
        has_vesicle_size=has_vesicle_size,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors
    )
    
    print(f"All projections completed successfully! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()