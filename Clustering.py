import kaleido
import plotly

import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Import from newdl module instead of synapse
from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset2 import SynapseDataset
from synapse import config

from synapse.clustering import (
    load_and_cluster_features, 
    apply_tsne, 
    save_tsne_plots,
    save_cluster_samples, 
    find_random_samples_in_clusters
)

config.parse_args()

data_loader = SynapseDataLoader(
    raw_base_dir=config.raw_base_dir,
    seg_base_dir=config.seg_base_dir,
    add_mask_base_dir=config.add_mask_base_dir
)

vol_data_dict = {}
for bbox_name in config.bbox_name:
    raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
    if raw_vol is not None:
        vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)

syn_df = pd.concat([
    pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
    for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
])

processor = Synapse3DProcessor(size=config.size)
# Disable normalization for consistent gray values
processor.normalize_volume = False

dataset = SynapseDataset(
    vol_data_dict=vol_data_dict,
    synapse_df=syn_df,
    processor=processor,
    segmentation_type=config.segmentation_type,
    subvol_size=config.subvol_size,
    num_frames=config.num_frames,
    alpha=config.alpha,
    # No need to set normalize_across_volume as it defaults to False in the new version
    smart_crop=False,
    presynapse_weight=0.5,
    normalize_presynapse_size=False
)

color_mapping = {
    'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
    'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
}

csv_dir = os.path.join('results', 'csv_outputs')
csv_pattern = os.path.join(csv_dir, 'features_seg*_alpha*.csv')
import glob
csv_files = glob.glob(csv_pattern)

if not csv_files:
    print(f"No CSV files found matching pattern {csv_pattern}")
else:
    print(f"Found {len(csv_files)} CSV files for analysis")
    for csv in csv_files:
        print(f"  - {os.path.basename(csv)}")

for csv_file in csv_files:
    csv_filepath = csv_file

    print(f"Processing {csv_file}")

    iteration_name = Path(csv_file).stem
    output_dir = Path(config.clustering_output_dir) / iteration_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the clustering algorithm and parameters from the config
    features_df, clusterer, feature_cols = load_and_cluster_features(
        csv_filepath, 
        n_clusters=config.n_clusters, 
        clustering_algorithm=config.clustering_algorithm,
        dbscan_eps=config.dbscan_eps, 
        dbscan_min_samples=config.dbscan_min_samples
    )

    features_df.to_csv(output_dir / "clustered_features.csv", index=False)

    tsne_results_2d = apply_tsne(features_df, feature_cols, 2)
    tsne_results_3d = apply_tsne(features_df, feature_cols, 3)

    save_tsne_plots(features_df, tsne_results_2d, tsne_results_3d, clusterer, color_mapping, output_dir)

    random_samples_per_cluster = find_random_samples_in_clusters(features_df, feature_cols, 4)

    save_cluster_samples(dataset, random_samples_per_cluster, output_dir)

    print(f"Saved results to {output_dir}")
