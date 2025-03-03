#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SynapseClusterEM Configuration Generator
========================================

This script helps users generate configuration files for the SynapseClusterEM project.
It provides an interactive command-line interface to set up the configuration parameters.

Usage:
    python generate_config.py --output config.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate SynapseClusterEM configuration file")
    parser.add_argument('--output', type=str, default='config.yaml',
                        help='Output path for the configuration file')
    parser.add_argument('--template', action='store_true',
                        help='Generate a template configuration file with default values')
    return parser.parse_args()

def get_input(prompt, default=None, options=None, input_type=str, required=False):
    """Get user input with validation"""
    default_str = f" [{default}]" if default is not None else ""
    options_str = f" (options: {', '.join(map(str, options))})" if options else ""
    
    while True:
        user_input = input(f"{prompt}{default_str}{options_str}: ").strip()
        
        if not user_input and default is not None:
            return default
        
        if not user_input and required:
            print("This field is required. Please provide a value.")
            continue
        
        if not user_input and not required:
            return None
        
        if options and user_input not in map(str, options):
            print(f"Invalid input. Please choose from: {', '.join(map(str, options))}")
            continue
        
        try:
            if input_type == bool:
                return user_input.lower() in ('yes', 'y', 'true', 't', '1')
            elif input_type == list:
                if not user_input:
                    return []
                return [item.strip() for item in user_input.split(',')]
            elif input_type == int:
                return int(user_input)
            elif input_type == float:
                return float(user_input)
            else:
                return user_input
        except ValueError:
            print(f"Invalid input. Expected type: {input_type.__name__}")

def generate_template_config():
    """Generate a template configuration file with default values"""
    config = {
        'mode': 'all',
        'data_paths': {
            'raw_base_dir': '/path/to/raw/data',
            'seg_base_dir': '/path/to/seg/data',
            'add_mask_base_dir': '',
            'excel_dir': '/path/to/excel/files',
            'output_dir': 'outputs/default',
            'checkpoint_path': '/path/to/vgg3d_checkpoint.pth',
        },
        'dataset': {
            'bbox_names': ['bbox1', 'bbox2', 'bbox3'],
            'size': [80, 80],
            'subvol_size': 80,
            'num_frames': 80,
        },
        'analysis': {
            'segmentation_types': [9, 10],
            'alphas': [1.0],
            'n_clusters': 10,
            'clustering_method': 'kmeans',
            'batch_size': 2,
            'num_workers': 0,
        },
        'normalization': {
            'use_global_norm': False,
            'global_stats_path': None,
            'num_samples_for_stats': 100,
        },
        'visualization': {
            'create_3d_plots': False,
            'save_interactive': False,
        },
        'system': {
            'gpu_id': 0,
            'seed': 42,
            'verbose': True,
        }
    }
    
    return config

def generate_interactive_config():
    """Generate a configuration file interactively"""
    config = {}
    
    print("\n=== SynapseClusterEM Configuration Generator ===\n")
    print("This tool will help you create a configuration file for the SynapseClusterEM project.")
    print("Press Enter to accept default values (shown in brackets).\n")
    
    # Workflow control
    print("\n--- Workflow Control ---")
    config['mode'] = get_input(
        "Pipeline mode", 
        default='all', 
        options=['preprocess', 'extract', 'cluster', 'visualize', 'all']
    )
    
    # Data paths
    print("\n--- Data Paths ---")
    data_paths = {}
    data_paths['raw_base_dir'] = get_input("Raw data directory", required=True)
    data_paths['seg_base_dir'] = get_input("Segmentation data directory", required=True)
    data_paths['add_mask_base_dir'] = get_input("Additional mask directory (optional)")
    data_paths['excel_dir'] = get_input("Excel files directory", required=True)
    data_paths['output_dir'] = get_input("Output directory", default='outputs/default')
    data_paths['checkpoint_path'] = get_input("VGG3D model checkpoint path", required=True)
    config['data_paths'] = data_paths
    
    # Dataset parameters
    print("\n--- Dataset Parameters ---")
    dataset = {}
    bbox_input = get_input("Bounding box names (comma-separated)", default='bbox1', input_type=list)
    dataset['bbox_names'] = bbox_input
    
    size_h = get_input("2D slice height", default=80, input_type=int)
    size_w = get_input("2D slice width", default=80, input_type=int)
    dataset['size'] = [size_h, size_w]
    
    dataset['subvol_size'] = get_input("3D subvolume size", default=80, input_type=int)
    dataset['num_frames'] = get_input("Number of frames", default=80, input_type=int)
    config['dataset'] = dataset
    
    # Analysis parameters
    print("\n--- Analysis Parameters ---")
    analysis = {}
    seg_types_input = get_input("Segmentation types (comma-separated integers)", default='9,10', input_type=str)
    analysis['segmentation_types'] = [int(x.strip()) for x in seg_types_input.split(',')]
    
    alphas_input = get_input("Alpha values (comma-separated floats)", default='1.0', input_type=str)
    analysis['alphas'] = [float(x.strip()) for x in alphas_input.split(',')]
    
    analysis['n_clusters'] = get_input("Number of clusters", default=10, input_type=int)
    analysis['clustering_method'] = get_input(
        "Clustering method", 
        default='kmeans', 
        options=['kmeans', 'dbscan']
    )
    analysis['batch_size'] = get_input("Batch size", default=2, input_type=int)
    analysis['num_workers'] = get_input("Number of workers", default=0, input_type=int)
    config['analysis'] = analysis
    
    # Global normalization parameters
    print("\n--- Global Normalization Parameters ---")
    normalization = {}
    normalization['use_global_norm'] = get_input("Use global normalization", default=False, input_type=bool)
    
    if normalization['use_global_norm']:
        normalization['global_stats_path'] = get_input("Path to global stats JSON (leave empty to calculate)")
        normalization['num_samples_for_stats'] = get_input(
            "Number of samples for global stats (0 for all)", 
            default=100, 
            input_type=int
        )
    else:
        normalization['global_stats_path'] = None
        normalization['num_samples_for_stats'] = 100
    
    config['normalization'] = normalization
    
    # Visualization parameters
    print("\n--- Visualization Parameters ---")
    visualization = {}
    visualization['create_3d_plots'] = get_input("Create 3D plots", default=False, input_type=bool)
    visualization['save_interactive'] = get_input("Save interactive HTML visualizations", default=False, input_type=bool)
    config['visualization'] = visualization
    
    # System parameters
    print("\n--- System Parameters ---")
    system = {}
    system['gpu_id'] = get_input("GPU ID (-1 for CPU)", default=0, input_type=int)
    system['seed'] = get_input("Random seed", default=42, input_type=int)
    system['verbose'] = get_input("Enable verbose logging", default=True, input_type=bool)
    config['system'] = system
    
    return config

def main():
    """Main function"""
    args = parse_args()
    
    if args.template:
        config = generate_template_config()
        print(f"Generating template configuration file: {args.output}")
    else:
        config = generate_interactive_config()
        print(f"\nGenerating configuration file: {args.output}")
    
    # Save configuration to file
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration file saved to: {args.output}")
    print("\nYou can now run the analysis with:")
    print(f"python main.py --config {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 