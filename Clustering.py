import kaleido #required
import plotly

import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Import modularized components
from synapse import (
    SynapseDataLoader, 
    Synapse3DProcessor, 
    SynapseDataset, 
    config
)
from synapse.utils.clusterhelper import (
    load_and_cluster_features, 
    apply_tsne, 
    find_closest_samples_in_clusters,
    save_tsne_plots,
    save_cluster_samples, 
    find_random_samples_in_clusters
)

# Define bounding box names to process
bbox_names = ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']

# Initialize and parse configuration
config.parse_args()

# Override specific configurations if needed
config.bbox_name = bbox_names
config.segmentation_type = 5

# Initialize data loader with configuration
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

# Initialize processor
processor = Synapse3DProcessor(size=config.size)

# Create dataset and features
dataset = SynapseDataset(
    vol_data_dict=vol_data_dict,
    synapse_df=syn_df,
    processor=processor,
    segmentation_type=config.segmentation_type,
    subvol_size=config.subvol_size,
    num_frames=config.num_frames,
    alpha=config.alpha
)

# Color mapping for visualization
color_mapping = {
    'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
    'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
}

# Get CSV file paths using the config helper
csv_files = [
    # os.path.join('csv_outputs', 'features_seg9_alpha1.csv'),
    os.path.join('csv_outputs', 'features_seg10_alpha1.csv')
]

# Process each CSV file
for csv_file in csv_files:
    csv_filepath = csv_file

    print(f"Processing {csv_file}")

    # Create output directory for this iteration
    iteration_name = Path(csv_file).stem  # Remove .csv extension
    output_dir = Path(config.clustering_output_dir) / iteration_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and cluster features
    n_clusters = 10
    features_df, kmeans, feature_cols = load_and_cluster_features(csv_filepath, n_clusters)

    # Save clustered features
    features_df.to_csv(output_dir / "clustered_features.csv", index=False)

    # Step 2: Apply t-SNE
    tsne_results_2d = apply_tsne(features_df, feature_cols, 2)
    tsne_results_3d = apply_tsne(features_df, feature_cols, 3)

    # Step 3: Save plots
    save_tsne_plots(features_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping, output_dir)

    # Step 4: Find and save closest samples
    # closest_samples_per_cluster = find_closest_samples_in_clusters(features_df, feature_cols, 4)
     # Step 4: Find and save random samples
    random_samples_per_cluster = find_random_samples_in_clusters(features_df, feature_cols, 4)

    # Step 5: Save cluster samples as images
    save_cluster_samples(dataset, random_samples_per_cluster, output_dir)

    print(f"Saved results to {output_dir}")

