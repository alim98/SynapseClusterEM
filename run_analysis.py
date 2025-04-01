#!/usr/bin/env python
"""
Synapse Analysis - Main Runner Script

This script provides a simplified interface to run the synapse analysis pipeline.
It handles configuration and runs the main analysis functions.
"""

import os
import argparse
import torch
import pandas as pd
# Import from the reorganized modules
from synapse import config
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from synapse import Vgg3D, load_model_from_checkpoint
from inference import run_full_analysis, load_and_prepare_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Synapse Analysis Pipeline")
    
    # Basic configuration
    parser.add_argument('--bbox_name', type=str, nargs='+', default=['bbox1'],
                      help='Bounding box names to analyze')
    parser.add_argument('--segmentation_type', type=int, default=6,
                      help='Segmentation type to use')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Alpha value for visualization')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Base directory for data')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--results_dir', type=str, default=None,
                      help='Explicit directory for results (overrides output_dir/csv_outputs)')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, default='hemibrain_production.checkpoint',
                      help='Path to model checkpoint')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update config with command-line arguments
    config.bbox_name = args.bbox_name
    config.segmentation_type = args.segmentation_type
    config.alpha = args.alpha
    
    # Set data paths based on data_dir argument
    config.raw_base_dir = os.path.join(args.data_dir, '7_bboxes_plus_seg/raw')
    config.seg_base_dir = os.path.join(args.data_dir, '7_bboxes_plus_seg/seg')
    config.add_mask_base_dir = os.path.join(args.data_dir, 'vesicle_cloud__syn_interface__mitochondria_annotation')
    config.excel_file = os.path.join(args.data_dir, '7_bboxes_plus_seg')
    
    # Set output directory
    config.csv_output_dir = os.path.join(args.output_dir, 'csv_outputs')
    
    # If results_dir is explicitly provided, use it for the results
    if args.results_dir:
        config.results_dir = args.results_dir
    else:
        # Otherwise, use the output_dir as the results base directory
        config.results_dir = args.output_dir
    
    os.makedirs(config.csv_output_dir, exist_ok=True)
    
    print(f"Running analysis with the following configuration:")
    print(f"  Bounding boxes: {config.bbox_name}")
    print(f"  Segmentation type: {config.segmentation_type}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Results will be saved in: {config.results_dir}/run_TIMESTAMP")
    print()
    
    # Initialize model
    print("Loading model...")
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, args.checkpoint_path)
    
    # Load and prepare data
    print("Loading data...")
    vol_data_dict, syn_df = load_and_prepare_data(config)
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    
    # Run analysis
    print("Starting analysis...")
    run_full_analysis(config, vol_data_dict, syn_df, processor, model)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 