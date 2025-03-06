import os
import glob
import io
from typing import List, Tuple
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms
from scipy.ndimage import label
from scipy import ndimage
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import umap
import shutil
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import kaleido
from pathlib import Path

# Import from the reorganized modules
from synapse import (
    Synapse3DProcessor, 
    SynapseDataLoader, 
    Vgg3D, 
    load_model_from_checkpoint,
    SynapseDataset,
    SynapseDataset2,
    config
)

# Import clustering functions
from synapse.utils.clusterhelper import (
    load_and_cluster_features,
    apply_tsne,
    find_closest_samples_in_clusters,
    save_tsne_plots,
    save_cluster_samples
)

# Add unique IDs to fixed_samples
fixed_samples = [
    {"id": 1, "bbox_name": "bbox1", "Var1": "non_spine_synapse_004", "slice_number": 25},
    {"id": 2, "bbox_name": "bbox1", "Var1": "non_spine_synapse_006", "slice_number": 40},
    {"id": 4, "bbox_name": "bbox2", "Var1": "explorative_2024-08-28_Cora_Wolter_031", "slice_number": 43},
    {"id": 3, "bbox_name": "bbox2", "Var1": "explorative_2024-08-28_Cora_Wolter_051", "slice_number": 28},
    {"id": 5, "bbox_name": "bbox3", "Var1": "non_spine_synapse_036", "slice_number": 41},
    {"id": 6, "bbox_name": "bbox3", "Var1": "non_spine_synapse_018", "slice_number": 41},
    {"id": 7, "bbox_name": "bbox4", "Var1": "explorative_2024-08-03_Ali_Karimi_023", "slice_number": 28},
    {"id": 8, "bbox_name": "bbox5", "Var1": "non_spine_synapse_033", "slice_number": 48},
    {"id": 9, "bbox_name": "bbox5", "Var1": "non_spine_synapse_045", "slice_number": 40},
    {"id": 10, "bbox_name": "bbox6", "Var1": "spine_synapse_070", "slice_number": 37},
    {"id": 11, "bbox_name": "bbox6", "Var1": "spine_synapse_021", "slice_number": 30},
    {"id": 12, "bbox_name": "bbox7", "Var1": "non_spine_synapse_013", "slice_number": 25},
]

def extract_features(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b]),  # Pixel values
            [item[1] for item in b],               # Synapse info
            [item[2] for item in b]                # Bbox names
        )
    )

    features = []
    metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", unit="batch"):
            pixels, info, names = batch
            inputs = pixels.permute(0, 2, 1, 3, 4).to(device)  # Reshape for 3D convolution

            batch_features = model.features(inputs)
            pooled_features = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)

            # Flatten but ensure consistent dimensions 
            # Don't squeeze unconditionally as it can remove too many dimensions
            batch_features_np = pooled_features.cpu().numpy()
            # Reshape to ensure consistent dimensions (batch_size, features)
            batch_size = batch_features_np.shape[0]
            num_features = np.prod(batch_features_np.shape[1:])
            batch_features_np = batch_features_np.reshape(batch_size, num_features)
            
            features.append(batch_features_np)
            metadata.extend(zip(names, info))

    # Combine all batch features
    features = np.concatenate(features, axis=0)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in metadata
    ])

    # Create feature columns
    feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)

    # Combine with metadata
    combined_df = pd.concat([metadata_df, features_df], axis=1)

    return combined_df

def create_plots(features_df, seg_type, alpha, fixed_samples):
    # Segmentation type descriptions
    print("Starting create_plots function")
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
    print("Processing features for UMAP")
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    features = features_df[feature_cols].values
    print(f"Features shape: {features.shape}")
    features_scaled = StandardScaler().fit_transform(features)
    print("Features scaled")

    # Compute UMAP only once
    print("Computing UMAP")
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)
    print(f"UMAP results shape: {umap_results.shape}")

    # Add UMAP coordinates to DataFrame
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    print("Added UMAP coordinates to DataFrame")

    # Fetch the description for the given seg_type
    seg_description = seg_type_descriptions.get(seg_type, "Unknown segmentation type")
    print(f"Segmentation description: {seg_description}")

    # Create Plotly figure
    print("Creating Plotly figure")
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
    print("Plotly figure created")

    # Use pandas merge for batch annotations
    print("Adding fixed samples")
    # Add cross markers for fixed samples
    fixed_samples_df = pd.DataFrame(fixed_samples)
    try:
        print("Merging dataframes")
        print(f"features_df columns: {features_df.columns}")
        print(f"fixed_samples_df columns: {fixed_samples_df.columns}")
        merged_df = features_df.reset_index().merge(fixed_samples_df, on=['Var1', 'bbox_name'])
        print(f"Merged DataFrame shape: {merged_df.shape}")

        # Add X markers
        print("Adding X markers")
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
        print("Added X markers")

        # Add annotations
        print("Adding annotations")
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
        print("Added annotations")
    except Exception as e:
        print(f"Error processing fixed samples: {str(e)}")
        import traceback
        traceback.print_exc()

    print("Returning figure from create_plots")
    return fig

def extract_and_save_features(model, dataset, config, seg_type, alpha, output_dir):
    print("Extracting features for SegType", seg_type, "and Alpha", alpha)
    features_df = extract_features(model, dataset, config)
    print("Features extracted for SegType", seg_type, "and Alpha", alpha)
    # Prepare the filename for CSV
    alpha_str = str(alpha).replace('.', '_')
    csv_filename = f"features_seg{seg_type}_alpha{alpha_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    print("Saving features to", csv_filepath)
    features_df.to_csv(csv_filepath, index=False)
    
    # Process features with UMAP for visualization
    print("Processing features for UMAP")
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    features = features_df[feature_cols].values
    print(f"Features shape: {features.shape}")
    features_scaled = StandardScaler().fit_transform(features)
    print("Features scaled")
    
    # Apply UMAP
    print("Computing UMAP")
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)
    print(f"UMAP results shape: {umap_results.shape}")
    
    # Add UMAP coordinates to DataFrame
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    print("Added UMAP coordinates to DataFrame")
    
    # Add segmentation type and alpha to the features DataFrame for tracking
    features_df['segmentation_type'] = seg_type
    features_df['alpha'] = alpha
    
    # Save updated features with UMAP coordinates
    features_df.to_csv(csv_filepath, index=False)
    print(f"Updated features saved to {csv_filepath}")
    
    # Perform clustering analysis (if clusterhelper is available)
    try:
        # Create segmentation-specific output directory
        seg_output_dir = os.path.join(output_dir, f"seg{seg_type}_alpha{alpha_str}")
        os.makedirs(seg_output_dir, exist_ok=True)
        
        # Load the features and perform clustering
        clustered_df, kmeans, feature_cols = load_and_cluster_features(csv_filepath, n_clusters=10)
        clustered_csv_path = os.path.join(seg_output_dir, "clustered_features.csv")
        clustered_df.to_csv(clustered_csv_path, index=False)
        
        # Perform t-SNE dimensionality reduction
        tsne_results_2d = apply_tsne(clustered_df, feature_cols, 2)
        tsne_results_3d = apply_tsne(clustered_df, feature_cols, 3)
        
        # Color mapping for visualization
        color_mapping = {
            'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
            'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
        }
        
        # Save t-SNE visualizations
        save_tsne_plots(clustered_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping, seg_output_dir)
        
        # Save cluster sample visualizations if dataset is provided
        closest_samples_per_cluster = find_closest_samples_in_clusters(clustered_df, feature_cols, 4)
        save_cluster_samples(dataset, closest_samples_per_cluster, seg_output_dir)
        
        print(f"Clustering analysis completed and saved to {seg_output_dir}")
    except Exception as e:
        print(f"Error during clustering analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return csv_filepath

def run_full_analysis(config, vol_data_dict, syn_df, processor, model):
    output_dir = config.csv_output_dir  # Directory to save CSVs
    
    segmentation_types = [1]  # Different segmentation types to analyze
    alpha_values = [ 1.0]  # Different alpha values to analyze
    combined_features = []
    
    # Feature extraction for different segmentation types and alpha values
    for seg_type in segmentation_types:
        for alpha in alpha_values:
            print(f"\n{'='*80}\nAnalyzing segmentation type {seg_type} with alpha {alpha}\n{'='*80}")
            
            # Create dataset with current segmentation type and alpha
            current_dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=syn_df,
                processor=processor,
                segmentation_type=seg_type,
                subvol_size=config.subvol_size,
                num_frames=config.num_frames,
                alpha=alpha
            )
            print("Dataset created.")
            
            # Extract features, save to CSV, and perform clustering
            csv_filepath = extract_and_save_features(model, current_dataset, config, seg_type, alpha, output_dir)
            
            # Load features from CSV to add to combined features
            features_df = pd.read_csv(csv_filepath)
            combined_features.append(features_df)
            
            # Create a dataset with only the fixed samples for comparison
            fixed_dataset = SynapseDataset2(
                vol_data_dict=vol_data_dict,
                synapse_df=syn_df,
                processor=processor,
                segmentation_type=seg_type,
                subvol_size=config.subvol_size,
                num_frames=config.num_frames,
                alpha=alpha,
                fixed_samples=fixed_samples  # Pass fixed_samples here
            )
            
            # Feature extraction for fixed samples
            for sample in fixed_samples:
                if sample["bbox_name"] in vol_data_dict:
                    print(f"Creating single sample visualization for {sample['Var1']} from {sample['bbox_name']}")
            
    # Run clustering on combined features from all segmentation types
    if combined_features:
        try:
            all_features_df = pd.concat(combined_features, ignore_index=True)
            combined_dir = os.path.join(output_dir, "combined_analysis")
            os.makedirs(combined_dir, exist_ok=True)
            all_features_csv = os.path.join(combined_dir, "all_features.csv")
            all_features_df.to_csv(all_features_csv, index=False)
            
            print("\n\nRunning clustering analysis on combined features from all segmentation types...")
            
            # Perform clustering on combined features
            clustered_df, kmeans, feature_cols = load_and_cluster_features(all_features_csv, n_clusters=10)
            clustered_csv_path = os.path.join(combined_dir, "clustered_features.csv")
            clustered_df.to_csv(clustered_csv_path, index=False)
            
            # Perform t-SNE on combined features
            tsne_results_2d = apply_tsne(clustered_df, feature_cols, 2)
            tsne_results_3d = apply_tsne(clustered_df, feature_cols, 3)
            
            # Color mapping for visualization
            color_mapping = {
                'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
                'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
            }
            
            # Save t-SNE visualizations for combined features
            save_tsne_plots(clustered_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping, combined_dir)
            print(f"Combined clustering analysis completed and saved to {combined_dir}")
        except Exception as e:
            print(f"Error during combined clustering analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("Analysis complete!")

# Load and prepare data
def load_and_prepare_data(config):
    """Load and prepare data using the configuration"""
    # Initialize data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    vol_data_dict = {}
    for bbox_name in config.bbox_name:
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)

    # Load synapse data
    syn_df = pd.concat([
        pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
        for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
    ])
    
    return vol_data_dict, syn_df

if __name__ == '__main__':
    checkpoint_url = "https://dl.dropboxusercontent.com/scl/fo/mfejaomhu43aa6oqs6zsf/AKMAAgT7OrUtruR0AQXZBy0/hemibrain_production.checkpoint.20220225?rlkey=6cmwxdvehy4ylztvsbgkfnrfc&dl=0"
    checkpoint_path = 'hemibrain_production.checkpoint'

    # Download the checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.system(f"wget -O {checkpoint_path} '{checkpoint_url}'")
        print("Downloaded VGG3D checkpoint.")
    else:
        print("VGG3D checkpoint already exists.")

    # Initialize and parse configuration
    config.parse_args()
    
    # Override specific configuration if needed
    config.bbox_name = ['bbox1']

    # Initialize model
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)

    # Load and prepare data
    vol_data_dict, syn_df = load_and_prepare_data(config)

    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)

    # Run the analysis
    run_full_analysis(config, vol_data_dict, syn_df, processor, model)
