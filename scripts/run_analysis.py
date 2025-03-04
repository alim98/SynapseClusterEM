import os
import argparse
import json
from pathlib import Path
import torch
import pandas as pd

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

def parse_args():
    parser = argparse.ArgumentParser(description="Synapse Analysis Pipeline")
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, default='raw')
    parser.add_argument('--seg_base_dir', type=str, default='seg')
    parser.add_argument('--add_mask_base_dir', type=str, default='')
    parser.add_argument('--excel_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    
    # Dataset parameters
    parser.add_argument('--bbox_names', type=str, nargs='+', default=['bbox1'])
    parser.add_argument('--size', type=int, nargs=2, default=[80, 80])
    parser.add_argument('--subvol_size', type=int, default=80)
    parser.add_argument('--num_frames', type=int, default=80)
    
    # Analysis parameters
    parser.add_argument('--segmentation_types', type=int, nargs='+', default=[9, 10])
    parser.add_argument('--alphas', type=float, nargs='+', default=[1.0])
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Global normalization parameters
    parser.add_argument('--use_global_norm', action='store_true', help='Use global normalization')
    parser.add_argument('--global_stats_path', type=str, help='Path to saved global stats JSON (will calculate if not provided)')
    parser.add_argument('--num_samples_for_stats', type=int, default=100, help='Number of samples for global stats (0 for all)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, args.checkpoint_path)
    
    # Load data
    vol_data_dict = load_all_volumes(
        args.bbox_names,
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    
    synapse_df = load_synapse_data(args.bbox_names, args.excel_dir)
    
    # Initialize processor based on normalization preference
    if args.use_global_norm:
        if args.global_stats_path and os.path.exists(args.global_stats_path):
            # Load pre-calculated global stats
            print(f"Loading global stats from {args.global_stats_path}")
            with open(args.global_stats_path, 'r') as f:
                global_stats = json.load(f)
                
            processor = Synapse3DProcessor(
                size=tuple(args.size),
                apply_global_norm=True,
                global_stats=global_stats
            )
            print(f"Using global normalization with mean={global_stats['mean']}, std={global_stats['std']}")
        else:
            # Calculate global stats
            print("Calculating global normalization statistics...")
            temp_processor = Synapse3DProcessor(size=tuple(args.size))
            temp_dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=synapse_df,
                processor=temp_processor
            )
            
            from torch.utils.data import DataLoader
            stats_loader = DataLoader(
                temp_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
            
            num_samples = args.num_samples_for_stats if args.num_samples_for_stats > 0 else None
            global_stats = Synapse3DProcessor.calculate_global_stats(stats_loader, num_samples=num_samples)
            
            # Save global stats for future use
            global_stats_path = output_dir / 'global_stats.json'
            with open(global_stats_path, 'w') as f:
                json.dump(global_stats, f)
            print(f"Global stats saved to {global_stats_path}")
            
            processor = Synapse3DProcessor(
                size=tuple(args.size),
                apply_global_norm=True,
                global_stats=global_stats
            )
            print(f"Using global normalization with mean={global_stats['mean']}, std={global_stats['std']}")
    else:
        # Use default normalization
        processor = Synapse3DProcessor(size=tuple(args.size))
        print("Using default normalization")
    
    # Color mapping for visualizations
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF',
        'bbox7': '#000000'
    }
    
    # Process each segmentation type and alpha value
    for seg_type in args.segmentation_types:
        for alpha in args.alphas:
            print(f"\nProcessing segmentation_type={seg_type}, alpha={alpha}")
            
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
            
            # Extract and save features
            norm_type = "global_norm" if args.use_global_norm else "default_norm"
            seg_output_dir = output_dir / f"seg{seg_type}_alpha{str(alpha).replace('.', '_')}_{norm_type}"
            seg_output_dir.mkdir(exist_ok=True)
            
            csv_filepath = extract_and_save_features(
                model=model,
                dataset=dataset,
                seg_type=seg_type,
                alpha=alpha,
                output_dir=seg_output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            # Load features and perform clustering
            features_df = pd.read_csv(csv_filepath)
            features_df, clustering_model = perform_clustering(
                features_df,
                method='kmeans',
                n_clusters=args.n_clusters
            )
            
            # Compute embeddings
            embeddings_2d = compute_embeddings(features_df, method='umap', n_components=2)
            embeddings_3d = compute_embeddings(features_df, method='umap', n_components=3)
            
            # Analyze clusters
            feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
            analyze_clusters(features_df, clustering_model, feature_cols, seg_output_dir)
            
            # Save visualizations
            save_cluster_visualizations(
                features_df,
                embeddings_2d,
                seg_output_dir,
                color_mapping
            )
            
            # Save 3D visualization
            save_cluster_visualizations(
                features_df,
                embeddings_3d,
                seg_output_dir / '3d',
                color_mapping
            )
            
            print(f"Results saved to {seg_output_dir}")

if __name__ == "__main__":
    main() 