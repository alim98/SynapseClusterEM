"""
Example script for using contrastive learning with the synapse pipeline.

This script demonstrates how to set up and run contrastive learning
for fine-tuning the VGG3D model used in the synapse pipeline.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import datetime

# Import from synapse pipeline
from synapse import config, Vgg3D, load_model_from_checkpoint
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from vgg3d_stage_extractor import VGG3DStageExtractor

# Import from contrastive learning package
from contrastive_learning import (
    ContrastiveLoss,
    SimCLRLoss,
    SynapseAugmenter,
    ContrastiveSynapseDataset,
    SupContrastiveSynapseDataset, 
    ContrastiveVGG3D,
    ContrastiveTrainer,
    simclr_loss
)

# Import pipeline integration functions
from contrastive_learning.pipeline import (
    load_model_for_contrastive,
    create_contrastive_datasets,
    setup_contrastive_training,
    train_with_contrastive_learning,
    add_contrastive_args_to_config
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run contrastive learning for synapse pipeline")
    
    # Basic arguments
    parser.add_argument("--use_contrastive_learning", action="store_true",
                      help="Use contrastive learning to fine-tune the model")
    parser.add_argument("--supervised_contrastive", action="store_true",
                      help="Use supervised contrastive learning with manual cluster annotations")
    parser.add_argument("--contrastive_epochs", type=int, default=20,
                      help="Number of epochs for contrastive learning")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate for contrastive training")
    
    # Data arguments
    parser.add_argument("--bbox_name", nargs="+", default=["7048_15"],
                      help="Names of bounding boxes to use")
    parser.add_argument("--raw_base_dir", type=str, default="data/raw",
                      help="Base directory for raw data")
    parser.add_argument("--seg_base_dir", type=str, default="data/segmentation",
                      help="Base directory for segmentation data")
    parser.add_argument("--excel_file", type=str, default="data/excel",
                      help="Directory containing excel files with synapse data")
    
    # Model arguments
    parser.add_argument("--model_checkpoint", type=str, default="hemibrain_production.checkpoint",
                      help="Path to the model checkpoint")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/contrastive",
                      help="Directory to save results")
    
    # Parse arguments
    args = parser.parse_args()
    
    return args


def load_data(config):
    """
    Load synapse data using the dataloader
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (vol_data_dict, syn_df)
    """
    print("Loading synapse data...")
    
    # Create data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=getattr(config, 'add_mask_base_dir', None)
    )
    
    # Load volumes
    vol_data_dict = {}
    for bbox_name in config.bbox_name:
        print(f"Loading data for {bbox_name}...")
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
    
    # Load synapse metadata from excel files
    syn_df_list = []
    for bbox in config.bbox_name:
        excel_path = os.path.join(config.excel_file, f"{bbox}.xlsx")
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            df['bbox_name'] = bbox
            syn_df_list.append(df)
    
    # Combine all dataframes
    if syn_df_list:
        syn_df = pd.concat(syn_df_list)
        print(f"Loaded {len(syn_df)} synapses from {len(syn_df_list)} excel files")
    else:
        print("No synapse data loaded from excel files")
        syn_df = None
    
    return vol_data_dict, syn_df


def load_manual_clusters(config):
    """
    Load manual cluster annotations if available
    
    Args:
        config: Configuration object
        
    Returns:
        pd.DataFrame: DataFrame with manual cluster annotations
    """
    print("Loading manual cluster annotations...")
    
    # Try multiple potential locations for manual clustering file
    potential_paths = [
        "manual_clustered_samples.csv",
        "manual/manual_clustered_samples.csv",
        "manual/clustering_results/manual_clustered_samples.csv",
        "clustered_samples.csv"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            manual_df = pd.read_csv(path)
            print(f"Loaded manual clusters from {path}")
            return manual_df
    
    print("Manual cluster annotations not found")
    return None


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up configuration
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    # Add contrastive learning parameters to config
    config = add_contrastive_args_to_config(config)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir
    
    # Load data
    vol_data_dict, syn_df = load_data(config)
    
    if vol_data_dict is None or syn_df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Load manual cluster annotations if using supervised contrastive learning
    manual_clusters_df = None
    if config.supervised_contrastive:
        manual_clusters_df = load_manual_clusters(config)
        
        if manual_clusters_df is None:
            print("WARNING: Manual clusters not found, falling back to unsupervised contrastive learning")
            config.supervised_contrastive = False
    
    # Train with contrastive learning
    encoder, history = train_with_contrastive_learning(
        config=config,
        vol_data_dict=vol_data_dict,
        syn_df=syn_df,
        manual_clusters_df=manual_clusters_df
    )
    
    print("Contrastive learning completed!")
    
    # Extract features using the fine-tuned encoder
    print("Now you can use the fine-tuned model for feature extraction in your pipeline.")
    

if __name__ == "__main__":
    main() 