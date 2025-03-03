#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from synapse_analysis.models.vgg3d import Vgg3D, load_model_from_checkpoint
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data
)
from synapse_analysis.data.dataset import SynapseDataset
from synapse_analysis.analysis.feature_extraction import extract_features
from synapse_analysis.utils.processing import create_segmented_cube

def parse_args():
    parser = argparse.ArgumentParser(description="Run analysis from the notebook")
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, required=True)
    parser.add_argument('--seg_base_dir', type=str, required=True)
    parser.add_argument('--add_mask_base_dir', type=str, required=True)
    parser.add_argument('--excel_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--checkpoint_path', type=str, default='hemibrain_production.checkpoint')
    
    # Dataset parameters
    parser.add_argument('--bbox_names', type=str, nargs='+', default=['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7'])
    parser.add_argument('--size', type=int, nargs=2, default=[80, 80])
    parser.add_argument('--subvol_size', type=int, default=80)
    parser.add_argument('--num_frames', type=int, default=80)
    
    # Analysis parameters
    parser.add_argument('--segmentation_types', type=int, nargs='+', default=[9, 10])
    parser.add_argument('--alphas', type=float, nargs='+', default=[1.0])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_global_norm', action='store_true', help='Use global normalization')
    
    return parser.parse_args()

def create_plots(features_df, seg_type, alpha, fixed_samples=None):
    """
    Create UMAP plots from feature data.
    
    Args:
        features_df: DataFrame containing features
        seg_type: Segmentation type
        alpha: Alpha blending factor
        fixed_samples: List of fixed samples to highlight
        
    Returns:
        Plotly figure
    """
    # Segmentation type descriptions
    seg_type_descriptions = {
        0: "Raw data",
        1: "Presynapse",
        2: "Postsynapse",
        3: "Both sides",
        4: "Vesicles + Cleft (closest only)",
        5: "(closest vesicles/cleft + sides)",
        6: "Vesicle cloud (closest)",
        7: "Cleft (closest)",
        8: "Mitochondria (closest)",
        9: "Vesicle + Cleft",
        10: "Cleft + Pre"
    }

    # Process features and compute UMAP
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    features = features_df[feature_cols].values
    features_scaled = StandardScaler().fit_transform(features)

    # Compute UMAP
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)

    # Add UMAP coordinates to DataFrame
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]

    # Fetch the description for the given seg_type
    seg_description = seg_type_descriptions.get(seg_type, "Unknown segmentation type")

    # Create Plotly figure
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    fig = px.scatter(
        features_df.reset_index(),  # Reset index to make it available as a column
        x='umap_x',
        y='umap_y',
        color='bbox_name',
        title=f"VGG Segmentation Type {seg_type} ({seg_description}) (α={alpha})",
        color_discrete_map=color_mapping,
        hover_data=['index', 'bbox_name'],  # Show index and bbox in hover
        labels={'color': 'BBox'}
    )

    # Add cross markers for fixed samples if provided
    if fixed_samples is not None:
        fixed_samples_df = pd.DataFrame(fixed_samples)
        merged_df = features_df.reset_index().merge(fixed_samples_df, on=['Var1', 'bbox_name'])

        # Add X markers
        fig.add_trace(
            go.Scatter(
                x=merged_df['umap_x'],
                y=merged_df['umap_y'],
                mode='markers',
                marker=dict(
                    symbol='x',
                    color='black',
                    size=10,
                    line=dict(width=2)
                ),
                name='Selected Samples',
                hoverinfo='none'
            )
        )

        # Add annotations
        for _, row in merged_df.iterrows():
            fig.add_annotation(
                x=row['umap_x'],
                y=row['umap_y'],
                text=str(row['id']),
                showarrow=True,
                font=dict(color='black', size=30),
                arrowhead=2,
                arrowsize=1,
                arrowcolor='black',
                ax=20,
                ay=-30,
                axref='pixel',
                ayref='pixel',
                xshift=10,
                yshift=10,
                bordercolor='white',
                borderwidth=1
            )

    return fig

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download checkpoint if it doesn't exist
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        checkpoint_url = "https://dl.dropboxusercontent.com/scl/fo/mfejaomhu43aa6oqs6zsf/AKMAAgT7OrUtruR0AQXZBy0/hemibrain_production.checkpoint.20220225?rlkey=6cmwxdvehy4ylztvsbgkfnrfc&dl=0"
        os.system(f"wget -O {checkpoint_path} '{checkpoint_url}'")
        print("Downloaded VGG3D checkpoint.")
    else:
        print("VGG3D checkpoint already exists.")
    
    # Load model
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)
    
    # Load data
    vol_data_dict = load_all_volumes(
        args.bbox_names,
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    
    synapse_df = load_synapse_data(args.bbox_names, args.excel_dir)
    
    # Initialize processor
    processor = Synapse3DProcessor(size=tuple(args.size))
    
    # Process each segmentation type and alpha value
    for seg_type in args.segmentation_types:
        for alpha in args.alphas:
            print(f"Processing segmentation type {seg_type} with alpha {alpha}")
            
            # Create dataset
            dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=synapse_df,
                processor=processor,
                segmentation_type=seg_type,
                subvol_size=args.subvol_size,
                num_frames=args.num_frames,
                alpha=alpha
            )
            
            # Extract features
            features_df = extract_features(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            # Save features
            features_path = output_dir / f"features_seg{seg_type}_alpha{alpha}.csv"
            features_df.to_csv(features_path, index=False)
            print(f"Saved features to {features_path}")
            
            # Create and save UMAP plot
            fig = create_plots(features_df, seg_type, alpha)
            plot_path = output_dir / f"umap_seg{seg_type}_alpha{alpha}.html"
            fig.write_html(str(plot_path))
            print(f"Saved UMAP plot to {plot_path}")
            
            # Also save as PNG for easy viewing
            png_path = output_dir / f"umap_seg{seg_type}_alpha{alpha}.png"
            fig.write_image(str(png_path))
            print(f"Saved UMAP plot image to {png_path}")

if __name__ == "__main__":
    main() 