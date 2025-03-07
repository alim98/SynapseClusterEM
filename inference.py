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
import argparse
from sklearn.cluster import KMeans

from synapse import (
    Synapse3DProcessor, 
    SynapseDataLoader, 
    Vgg3D, 
    load_model_from_checkpoint,
    SynapseDataset,
    SynapseDataset2,
    config
)

from synapse.utils.clusterhelper import (
    load_and_cluster_features,
    apply_tsne,
    find_random_samples_in_clusters,
    save_tsne_plots,
    save_cluster_samples
)

from presynapse_analysis import run_presynapse_analysis

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
            torch.stack([item[0] for item in b]),
            [item[1] for item in b],
            [item[2] for item in b]
        )
    )

    features = []
    metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", unit="batch"):
            pixels, info, names = batch
            inputs = pixels.permute(0, 2, 1, 3, 4).to(device)

            batch_features = model.features(inputs)
            pooled_features = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)

            batch_features_np = pooled_features.cpu().numpy()
            batch_size = batch_features_np.shape[0]
            num_features = np.prod(batch_features_np.shape[1:])
            batch_features_np = batch_features_np.reshape(batch_size, num_features)
            
            features.append(batch_features_np)
            metadata.extend(zip(names, info))

    features = np.concatenate(features, axis=0)

    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in metadata
    ])

    feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)

    combined_df = pd.concat([metadata_df, features_df], axis=1)

    return combined_df

def create_plots(features_df, seg_type, alpha, fixed_samples):
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

    print("Processing features for UMAP")
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    features = features_df[feature_cols].values
    print(f"Features shape: {features.shape}")
    features_scaled = StandardScaler().fit_transform(features)
    print("Features scaled")

    print("Computing UMAP")
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)
    print(f"UMAP results shape: {umap_results.shape}")

    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    print("Added UMAP coordinates to DataFrame")

    seg_description = seg_type_descriptions.get(seg_type, "Unknown segmentation type")
    print(f"Segmentation description: {seg_description}")

    print("Creating Plotly figure")
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    fig = px.scatter(
        features_df.reset_index(),
        x='umap_x',
        y='umap_y',
        color='bbox_name',
        title=f"VGG Segmentation Type {seg_type} ({seg_description}) (α={alpha})",
        color_discrete_map=color_mapping,
        hover_data=['index', 'bbox_name'],
        labels={'color': 'BBox'}
    )
    print("Plotly figure created")

    print("Adding fixed samples")
    fixed_samples_df = pd.DataFrame(fixed_samples)
    try:
        print("Merging dataframes")
        print(f"features_df columns: {features_df.columns}")
        print(f"fixed_samples_df columns: {fixed_samples_df.columns}")
        merged_df = features_df.reset_index().merge(fixed_samples_df, on=['Var1', 'bbox_name'])
        print(f"Merged DataFrame shape: {merged_df.shape}")

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
    alpha_str = str(alpha).replace('.', '_')
    csv_filename = f"features_seg{seg_type}_alpha{alpha_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    os.makedirs(output_dir, exist_ok=True)
    print("Saving features to", csv_filepath)
    features_df.to_csv(csv_filepath, index=False)
    
    print("Processing features for UMAP")
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    features = features_df[feature_cols].values
    print(f"Features shape: {features.shape}")
    features_scaled = StandardScaler().fit_transform(features)
    print("Features scaled")
    
    print("Computing UMAP")
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)
    print(f"UMAP results shape: {umap_results.shape}")
    
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    print("Added UMAP coordinates to DataFrame")
    
    features_df['segmentation_type'] = seg_type
    features_df['alpha'] = alpha
    
    features_df.to_csv(csv_filepath, index=False)
    print(f"Updated features saved to {csv_filepath}")
    
    try:
        seg_output_dir = os.path.join(output_dir, f"seg{seg_type}_alpha{alpha_str}")
        os.makedirs(seg_output_dir, exist_ok=True)
        
        clustered_df, kmeans, feature_cols = load_and_cluster_features(csv_filepath, n_clusters=10)
        clustered_csv_path = os.path.join(seg_output_dir, "clustered_features.csv")
        clustered_df.to_csv(clustered_csv_path, index=False)
        
        tsne_results_2d = apply_tsne(clustered_df, feature_cols, 2)
        tsne_results_3d = apply_tsne(clustered_df, feature_cols, 3)
        
        color_mapping = {
            'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
            'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
        }
        
        save_tsne_plots(clustered_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping, seg_output_dir)
        
        random_samples_in_clusters = find_random_samples_in_clusters(clustered_df, feature_cols, 4)
        save_cluster_samples(dataset, random_samples_in_clusters, seg_output_dir)
        
        print(f"Clustering analysis completed and saved to {seg_output_dir}")
    except Exception as e:
        print(f"Error during clustering analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return csv_filepath

def run_full_analysis(config, vol_data_dict, syn_df, processor, model):
    output_dir = config.csv_output_dir
    
    segmentation_types = [config.segmentation_type] if isinstance(config.segmentation_type, int) else config.segmentation_type
    alpha_values = [config.alpha] if isinstance(config.alpha, (int, float)) else config.alpha
    combined_features = []
    
    for seg_type in segmentation_types:
        for alpha in alpha_values:
            print(f"\n{'='*80}\nAnalyzing segmentation type {seg_type} with alpha {alpha}\n{'='*80}")
            
            current_dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=syn_df,
                processor=processor,
                segmentation_type=seg_type,
                alpha=alpha
            )
            
            # Step 1: Extract and save features if not skipped
            if not hasattr(config, 'skip_feature_extraction') or not config.skip_feature_extraction:
                features_path = extract_and_save_features(model, current_dataset, config, seg_type, alpha, output_dir)
            else:
                # Load existing features
                alpha_str = str(alpha).replace('.', '_')
                features_path = os.path.join(output_dir, f"features_seg{seg_type}_alpha{alpha_str}.csv")
                if not os.path.exists(features_path):
                    print(f"Error: Feature file not found at {features_path}")
                    print("Please run without --skip_feature_extraction first")
                    return
                print(f"Loading existing features from {features_path}")
            
            # Step 2: Perform clustering analysis if not skipped
            if not hasattr(config, 'skip_clustering') or not config.skip_clustering:
                try:
                    # Read features
                    features_df = pd.read_csv(features_path)
                    combined_features.append(features_df)
                    
                    # Add segmentation type and alpha as columns for the combined analysis
                    features_df['seg_type'] = seg_type
                    features_df['alpha'] = alpha
                    
                    # Convert alpha to string format for filenames
                    alpha_str = str(alpha).replace('.', '_')
                    
                    # Save sample visualizations
                    create_sample_visualizations(current_dataset, features_df, seg_type, alpha, output_dir)
                    
                    # Clustering analysis for this segmentation type and alpha
                    seg_output_dir = os.path.join(output_dir, f"seg{seg_type}_alpha{alpha_str}")
                    os.makedirs(seg_output_dir, exist_ok=True)
                    
                    # Perform UMAP and clustering
                    run_clustering_analysis(features_df, seg_output_dir)
                except Exception as e:
                    print(f"Error during analysis for seg_type={seg_type}, alpha={alpha}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    # Perform combined clustering analysis if not skipped and there are features
    if (not hasattr(config, 'skip_clustering') or not config.skip_clustering) and combined_features:
        print("\n\nRunning clustering analysis on combined features from all segmentation types...")
        try:
            # Combine all features
            combined_df = pd.concat(combined_features, ignore_index=True)
            
            # Save combined features
            combined_dir = os.path.join(output_dir, "combined_analysis")
            os.makedirs(combined_dir, exist_ok=True)
            combined_df.to_csv(os.path.join(combined_dir, "combined_features.csv"), index=False)
            
            # Run clustering on combined features
            run_clustering_analysis(combined_df, combined_dir)
            print(f"Combined clustering analysis completed and saved to {combined_dir}")
        except Exception as e:
            print(f"Error during combined clustering analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("Main analysis complete!")

    # Run presynapse analysis if not skipped
    if not hasattr(config, 'skip_presynapse_analysis') or not config.skip_presynapse_analysis:
        print("\n\n" + "="*80)
        print("Starting presynapse analysis to identify synapses with the same presynapse ID")
        print("="*80)
        run_presynapse_analysis(config)
    
    print("All analyses complete!")

    # Generate comprehensive reports if not skipped
    if not hasattr(config, 'skip_report_generation') or not config.skip_report_generation:
        try:
            from report_generator import SynapseReportGenerator
            
            print("\n\n" + "="*80)
            print("Generating comprehensive reports")
            print("="*80)
            
            report_generator = SynapseReportGenerator(
                csv_output_dir=config.csv_output_dir,
                clustering_output_dir=config.clustering_output_dir,
                report_output_dir=config.report_output_dir
            )
            
            # Generate reports
            comprehensive_report = report_generator.generate_complete_report()
            presynapse_summary = report_generator.generate_presynapse_summary()
            
            if comprehensive_report:
                print(f"Comprehensive report generated at: {comprehensive_report}")
            else:
                print("Failed to generate comprehensive report")
                
            if presynapse_summary:
                print(f"Presynapse summary report generated at: {presynapse_summary}")
            else:
                print("Failed to generate presynapse summary report")
        
        except Exception as e:
            print(f"Error generating reports: {e}")
            import traceback
            traceback.print_exc()
    
    print("Pipeline complete!")

def load_and_prepare_data(config):
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
    
    return vol_data_dict, syn_df

def run_clustering_analysis(features_df, output_dir):
    """
    Run clustering analysis on the features dataframe.
    
    Args:
        features_df: DataFrame with features
        output_dir: Directory to save results
    """
    # Extract feature columns
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    
    # Apply UMAP dimensionality reduction
    umap_results = apply_umap(features_df[feature_cols])
    features_df['umap_1'] = umap_results[:, 0]
    features_df['umap_2'] = umap_results[:, 1]
    
    # Apply clustering based on selected algorithm
    if config.clustering_algorithm == 'KMeans':
        clusterer = KMeans(n_clusters=config.n_clusters, random_state=42)
        features_df['cluster'] = clusterer.fit_predict(features_df[feature_cols])
    else:  # DBSCAN
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
        features_df['cluster'] = clusterer.fit_predict(features_df[feature_cols])
        
        # DBSCAN assigns -1 to noise points, which can cause issues for visualization
        # Let's assign noise points to cluster 999 for easier identification
        features_df.loc[features_df['cluster'] == -1, 'cluster'] = 999
    
    # Save clustered features
    features_df.to_csv(os.path.join(output_dir, "clustered_features.csv"), index=False)
    
    # Apply t-SNE for visualization
    tsne_results_2d = apply_tsne(features_df, feature_cols, 2)
    tsne_results_3d = apply_tsne(features_df, feature_cols, 3)
    
    # Create color mapping for bounding boxes
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    # Save t-SNE plots
    save_tsne_plots(features_df, tsne_results_2d, tsne_results_3d, clusterer, color_mapping, output_dir)
    
    # Find and save sample images from each cluster
    random_samples_in_clusters = find_random_samples_in_clusters(features_df, feature_cols, 4)
    
    return features_df

def create_sample_visualizations(dataset, features_df, seg_type, alpha, output_dir):
    """
    Create visualizations for specific samples in the dataset.
    
    Args:
        dataset: SynapseDataset instance
        features_df: DataFrame with features
        seg_type: Segmentation type
        alpha: Alpha value
        output_dir: Directory to save results
    """
    # Sample a few synapses for visualization
    if len(features_df) > 5:
        sample_indices = np.random.choice(len(features_df), 5, replace=False)
        sample_rows = features_df.iloc[sample_indices]
        
        for _, sample in sample_rows.iterrows():
            bbox_name = sample['bbox_name']
            synapse_id = sample['Var1']
            
            if bbox_name in dataset.vol_data_dict:
                print(f"Creating single sample visualization for {synapse_id} from {bbox_name}")
                # Implementation of visualization logic would go here
                # This is a placeholder since we don't have the full visualization function

def find_random_samples_in_clusters(features_df, feature_cols, n_samples=2):
    """
    Find random samples in each cluster for visualization.
    
    Args:
        features_df: DataFrame with features and cluster assignments
        feature_cols: List of feature column names
        n_samples: Number of samples to select per cluster
        
    Returns:
        dict: Dictionary mapping cluster IDs to sample rows
    """
    if 'cluster' not in features_df.columns:
        print("No cluster information in features DataFrame")
        return {}
    
    random_samples = {}
    for cluster_id in features_df['cluster'].unique():
        cluster_samples = features_df[features_df['cluster'] == cluster_id]
        if len(cluster_samples) > 0:
            if len(cluster_samples) <= n_samples:
                selected_samples = cluster_samples
            else:
                selected_indices = np.random.choice(len(cluster_samples), n_samples, replace=False)
                selected_samples = cluster_samples.iloc[selected_indices]
            
            random_samples[cluster_id] = selected_samples
    
    return random_samples

def save_cluster_samples(dataset, random_samples, output_dir):
    """
    Save visualizations of sample synapses from each cluster.
    
    Args:
        dataset: SynapseDataset instance
        random_samples: Dictionary mapping cluster IDs to sample rows
        output_dir: Directory to save results
    """
    # This is a placeholder function for saving sample visualizations
    # The actual implementation would depend on how you want to visualize the samples
    
    for cluster_id, samples in random_samples.items():
        # Implementation of cluster sample visualization would go here
        # This is just a placeholder
        print(f"Saving {len(samples)} sample visualizations for cluster {cluster_id}")

def VGG3D():
    """
    Initialize and load the VGG3D model.
    
    Returns:
        model: Loaded VGG3D model
    """
    checkpoint_url = "https://dl.dropboxusercontent.com/scl/fo/mfejaomhu43aa6oqs6zsf/AKMAAgT7OrUtruR0AQXZBy0/hemibrain_production.checkpoint.20220225?rlkey=6cmwxdvehy4ylztvsbgkfnrfc&dl=0"
    checkpoint_path = 'hemibrain_production.checkpoint'

    if not os.path.exists(checkpoint_path):
        os.system(f"wget -O {checkpoint_path} '{checkpoint_url}'")
        print("Downloaded VGG3D checkpoint.")
    else:
        print("VGG3D checkpoint already exists.")

    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)
    print("Model loaded from hemibrain_production.checkpoint")
    
    return model

def apply_umap(features):
    """
    Apply UMAP dimensionality reduction to features.
    
    Args:
        features: Feature matrix
        
    Returns:
        numpy.ndarray: UMAP embedding
    """
    print("Computing UMAP")
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    print(f"UMAP results shape: {embedding.shape}")
    return embedding

def main():
    config.parse_args()
    
    # Setup directories
    os.makedirs(config.csv_output_dir, exist_ok=True)
    os.makedirs(config.save_gifs_dir, exist_ok=True)
    
    # Initialize VGG3D model
    model = VGG3D()

    # Load and prepare data
    vol_data_dict, syn_df = load_and_prepare_data(config)

    processor = Synapse3DProcessor(size=config.size)

    # Run the full analysis pipeline
    run_full_analysis(config, vol_data_dict, syn_df, processor, model)

    print("All analyses complete!")

    # Generate comprehensive reports if not skipped
    if not hasattr(config, 'skip_report_generation') or not config.skip_report_generation:
        try:
            from report_generator import SynapseReportGenerator
            
            print("\n\n" + "="*80)
            print("Generating comprehensive reports")
            print("="*80)
            
            report_generator = SynapseReportGenerator(
                csv_output_dir=config.csv_output_dir,
                clustering_output_dir=config.clustering_output_dir,
                report_output_dir=config.report_output_dir
            )
            
            # Generate reports
            comprehensive_report = report_generator.generate_complete_report()
            presynapse_summary = report_generator.generate_presynapse_summary()
            
            if comprehensive_report:
                print(f"Comprehensive report generated at: {comprehensive_report}")
            else:
                print("Failed to generate comprehensive report")
                
            if presynapse_summary:
                print(f"Presynapse summary report generated at: {presynapse_summary}")
            else:
                print("Failed to generate presynapse summary report")
        
        except Exception as e:
            print(f"Error generating reports: {e}")
            import traceback
            traceback.print_exc()
    
    print("Pipeline complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the analysis pipeline')
    parser.add_argument('--segmentation_type', type=int, default=config.segmentation_type, 
                       choices=range(0, 11), help='Type of segmentation overlay (0-10)')
    parser.add_argument('--gray_color', type=float, default=config.gray_color,
                       help='Gray color value (0-1) for overlaying segmentation')
    
    # Add flags to skip parts of the analysis (for GUI integration)
    parser.add_argument('--skip_feature_extraction', action='store_true',
                       help='Skip feature extraction and load existing features')
    parser.add_argument('--skip_clustering', action='store_true',
                       help='Skip clustering analysis')
    parser.add_argument('--skip_presynapse_analysis', action='store_true',
                       help='Skip presynapse analysis')
    parser.add_argument('--skip_report_generation', action='store_true',
                       help='Skip report generation')
    
    args, _ = parser.parse_known_args()

    main()
