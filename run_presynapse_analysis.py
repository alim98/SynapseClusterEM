#!/usr/bin/env python
"""
Presynapse Analysis Runner Script

This script runs the presynapse analysis to identify synapses with the same presynapse ID
and report on their clusters and feature distances.
"""

import argparse
import os
from synapse import config
from presynapse_analysis import run_presynapse_analysis

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Presynapse Analysis")
    
    # Basic configuration
    parser.add_argument('--bbox_name', type=str, nargs='+', default=['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7'],
                      help='Bounding box names to analyze')
    parser.add_argument('--segmentation_type', type=int, default=1,
                      help='Segmentation type to use (1 for presynapse, 2 for postsynapse)')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='Alpha value for visualization')
    
    # Data paths
    parser.add_argument('--csv_output_dir', type=str, default='results/csv_outputs',
                      help='Directory containing feature CSV files')
    parser.add_argument('--clustering_output_dir', type=str, default='results/clustering_results_final',
                      help='Directory to save clustering results')
    
    return parser.parse_args()

def main():
    """Main function to run the presynapse analysis."""
    # Parse arguments
    args = parse_arguments()
    config.parse_args()
    
    # Update config with parsed arguments
    config.parse_args()
    config.bbox_name = args.bbox_name
    config.segmentation_type = args.segmentation_type
    config.alpha = args.alpha
    config.csv_output_dir = args.csv_output_dir
    config.clustering_output_dir = args.clustering_output_dir
    
    print(f"Running presynapse analysis with the following settings:")
    print(f"  Bounding boxes: {config.bbox_name}")
    print(f"  Segmentation type: {config.segmentation_type}")
    print(f"  Alpha: {config.alpha}")
    print(f"  CSV output directory: {config.csv_output_dir}")
    print(f"  Clustering output directory: {config.clustering_output_dir}")
    
    # Run the presynapse analysis
    run_presynapse_analysis(config)

if __name__ == "__main__":
    main() 