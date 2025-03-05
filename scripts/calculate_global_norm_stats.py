#!/usr/bin/env python3
"""
Script to compute global normalization statistics for synapse data.

This script uses the GlobalNormalizationCalculator to compute,
save, and optionally display global normalization statistics
for the raw data.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse_analysis.data.data_loader import GlobalNormalizationCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate global normalization statistics")
    
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="Base directory for raw data (overrides config)")
    parser.add_argument("--output_file", type=str, default="global_stats.json",
                        help="Output file to save statistics")
    parser.add_argument("--bbox_names", type=str, nargs="+",
                        help="List of bounding box names (overrides config)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed information")
    
    return parser.parse_args()


def main():
    """Main function to calculate global normalization statistics."""
    args = parse_args()
    
    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config, but allow command line arguments to override
    raw_base_dir = args.raw_dir or config.get("raw_base_dir")
    seg_base_dir = config.get("seg_base_dir")
    add_mask_base_dir = config.get("add_mask_base_dir")
    bbox_names = args.bbox_names or config.get("bbox_names")
    
    # Create output directory if it doesn't exist
    output_file = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Raw data directory: {raw_base_dir}")
    print(f"Bounding boxes: {bbox_names}")
    print(f"Output file: {output_file}")
    
    # Create and run the calculator
    calculator = GlobalNormalizationCalculator(
        raw_base_dir=raw_base_dir,
        output_file=output_file
    )
    
    # Compute statistics from volumes
    stats = calculator.compute_from_volumes(
        bbox_names=bbox_names,
        seg_base_dir=seg_base_dir,
        add_mask_base_dir=add_mask_base_dir,
        verbose=args.verbose
    )
    
    # Print summary
    print("\nComputed global normalization statistics:")
    mean, std = calculator.get_normalization_parameters()
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    print(f"\nStatistics saved to: {output_file}")
    print("\nYou can now use these statistics for global normalization by:")
    print(f"1. Setting 'use_global_norm: true' in your config")
    print(f"2. Setting 'global_stats_path: \"{output_file}\"' in your config")


if __name__ == "__main__":
    main() 