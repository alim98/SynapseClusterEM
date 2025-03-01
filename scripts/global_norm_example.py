#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor, 
    load_synapse_data, 
    load_all_volumes
)
from synapse_analysis.data.dataset import SynapseDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Global Normalization Example')
    parser.add_argument('--raw_base_dir', type=str, required=True, help='Base directory for raw data')
    parser.add_argument('--seg_base_dir', type=str, required=True, help='Base directory for segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, required=True, help='Base directory for additional mask data')
    parser.add_argument('--excel_dir', type=str, required=True, help='Directory containing Excel files')
    parser.add_argument('--output_dir', type=str, default='./outputs/global_norm', help='Output directory')
    parser.add_argument('--segmentation_type', type=int, default=1, help='Segmentation type (0, 1, 2, 3, 4, or 9)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define bounding box names
    bbox_names = [f"bbox{i}" for i in range(1, 8)]  # adjust as needed
    
    # Load synapse data
    print("Loading synapse data...")
    synapse_df = load_synapse_data(bbox_names, args.excel_dir)
    
    # Load volume data
    print("Loading volume data...")
    vol_data_dict = load_all_volumes(
        bbox_names, 
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    
    # Calculate global stats directly from volumes
    print("\nCalculating global statistics directly from volumes...")
    global_stats = Synapse3DProcessor.calculate_global_stats_from_volumes(vol_data_dict)
    print(f"Global mean: {global_stats['mean']}")
    print(f"Global std: {global_stats['std']}")
    
    # Save global stats
    stats_file = os.path.join(args.output_dir, 'global_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(global_stats, f)
    print(f"Global statistics saved to {stats_file}")
    
    # Create processor with global normalization
    processor = Synapse3DProcessor.create_with_global_norm_from_volumes(vol_data_dict)
    
    # Create dataset with global normalization
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=synapse_df,
        processor=processor,
        segmentation_type=args.segmentation_type
    )
    
    print(f"\nCreated dataset with {len(dataset)} samples using global normalization")
    print(f"Segmentation type: {args.segmentation_type}")
    print("\nGlobal normalization example completed successfully!")
    print("\nTo use global normalization in your code:")
    print("1. Load the global statistics from the JSON file")
    print("2. Create a processor with global normalization")
    print("3. Use the processor with your dataset")
    print("\nExample:")
    print("```python")
    print("import json")
    print("from synapse_analysis.data.data_loader import Synapse3DProcessor")
    print(f"with open('{stats_file}', 'r') as f:")
    print("    global_stats = json.load(f)")
    print("processor = Synapse3DProcessor(apply_global_norm=True, global_stats=global_stats)")
    print("```")

if __name__ == "__main__":
    main() 