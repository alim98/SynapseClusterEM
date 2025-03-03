#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SynapseClusterEM Example Script
===============================

This script demonstrates how to use the main.py script for a common workflow
in the SynapseClusterEM project. It runs the complete analysis pipeline with
typical parameters.

Usage:
    python run_synapse_analysis.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SynapseClusterEM Analysis")
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, required=True,
                        help='Directory containing raw image data')
    parser.add_argument('--seg_base_dir', type=str, required=True,
                        help='Directory containing segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, default='',
                        help='Directory containing additional mask data')
    parser.add_argument('--excel_dir', type=str, required=True,
                        help='Directory containing Excel files with synapse information')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the VGG3D model checkpoint')
    
    # Analysis parameters
    parser.add_argument('--output_dir', type=str, default='outputs/synapse_analysis',
                        help='Directory to save output files')
    parser.add_argument('--bbox_names', type=str, nargs='+', default=['bbox1'],
                        help='Names of bounding boxes to process')
    parser.add_argument('--segmentation_types', type=int, nargs='+', default=[9, 10],
                        help='Segmentation types to analyze')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for K-means')
    parser.add_argument('--use_global_norm', action='store_true',
                        help='Use global normalization')
    parser.add_argument('--create_3d_plots', action='store_true',
                        help='Create 3D visualizations')
    parser.add_argument('--save_interactive', action='store_true',
                        help='Save interactive HTML visualizations')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis pipeline"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build command for main.py
    cmd = [
        sys.executable,
        "main.py",
        "--raw_base_dir", args.raw_base_dir,
        "--seg_base_dir", args.seg_base_dir,
        "--excel_dir", args.excel_dir,
        "--checkpoint_path", args.checkpoint_path,
        "--output_dir", args.output_dir,
        "--n_clusters", str(args.n_clusters),
    ]
    
    # Add optional arguments
    if args.add_mask_base_dir:
        cmd.extend(["--add_mask_base_dir", args.add_mask_base_dir])
    
    if args.bbox_names:
        cmd.extend(["--bbox_names"] + args.bbox_names)
    
    if args.segmentation_types:
        cmd.extend(["--segmentation_types"] + [str(t) for t in args.segmentation_types])
    
    if args.use_global_norm:
        cmd.append("--use_global_norm")
    
    if args.create_3d_plots:
        cmd.append("--create_3d_plots")
    
    if args.save_interactive:
        cmd.append("--save_interactive")
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    print("\n")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("\nAnalysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError running analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 