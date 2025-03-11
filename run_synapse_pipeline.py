"""
Main script for running the synapse analysis pipeline.

This script demonstrates how to use the SynapsePipeline class
to orchestrate the entire synapse analysis workflow.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from glob import glob
import time
import datetime  # Added for timestamp

from synapse import config
from synapse_pipeline import SynapsePipeline
from vesicle_size_visualizer import (
    compute_vesicle_cloud_sizes,
    create_umap_with_vesicle_sizes,
    analyze_vesicle_sizes_by_cluster,
    count_bboxes_in_clusters,
    plot_bboxes_in_clusters
)


# Set up logging to a file
log_file = open("pipeline_log.txt", "a")

def log_print(*args, **kwargs):
    """Custom print function that prints to both console and log file"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)
    log_file.flush()  # Make sure it's written immediately


def configure_pipeline_args():
    """Configure pipeline arguments by extending the existing config object"""
    # Create a new parser
    parser = argparse.ArgumentParser(description="Run the synapse analysis pipeline")
    
    # Add pipeline-specific arguments
    parser.add_argument("--only_vesicle_analysis", action="store_true",
                        help="Only run vesicle size analysis on existing results")
    
    # Add feature extraction method arguments
    parser.add_argument("--extraction_method", type=str, choices=['standard', 'stage_specific'],
                       help="Method to extract features ('standard' or 'stage_specific')")
    parser.add_argument("--layer_num", type=int,
                       help="Layer number to extract features from when using stage_specific method")
    
    # Let config parse its arguments first
    config.parse_args()
    
    # Parse our additional arguments
    args, _ = parser.parse_known_args()
    
    # Add our arguments to config
    if args.only_vesicle_analysis:
        config.only_vesicle_analysis = True
    else:
        config.only_vesicle_analysis = False
    
    # Add feature extraction parameters if provided
    if args.extraction_method:
        config.extraction_method = args.extraction_method
    
    if args.layer_num:
        config.layer_num = args.layer_num
    
    return config


def run_vesicle_analysis():
    """
    Run vesicle analysis on existing features.
    This is a stub function that will be implemented with the full vesicle analysis logic.
    """
    log_print("Vesicle analysis functionality will be implemented soon.")
    pass


def analyze_vesicle_sizes(pipeline, features_df):
    """
    Analyze vesicle sizes using the provided features.
    
    Args:
        pipeline: SynapsePipeline instance
        features_df: DataFrame with extracted features
        
    Returns:
        dict: Analysis results
    """
    log_print("Vesicle size analysis functionality will be implemented soon.")
    return {"status": "success"}


def main():
    """Main function to run the pipeline"""
    log_print(f"\n--- Starting pipeline run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # Configure pipeline arguments
    configure_pipeline_args()
    
    # Log configuration details
    log_print(f"Running with configuration:")
    log_print(f"  Segmentation Type: {config.segmentation_type}")
    log_print(f"  Alpha: {config.alpha}")
    log_print(f"  Feature Extraction Method: {getattr(config, 'extraction_method', 'standard')}")
    if getattr(config, 'extraction_method', 'standard') == 'stage_specific':
        log_print(f"  Layer Number: {getattr(config, 'layer_num', 20)}")
    
    # Get feature extraction parameters
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hasattr(config, 'only_vesicle_analysis') and config.only_vesicle_analysis:
        log_print("Running only vesicle analysis on existing results...")
        run_vesicle_analysis()
    else:
        log_print("Running full pipeline...")
        
        # Initialize and run the pipeline
        pipeline = SynapsePipeline(config)
        result = pipeline.run_full_pipeline(
            seg_type=config.segmentation_type,
            alpha=config.alpha,
            extraction_method=extraction_method,
            layer_num=layer_num
        )
        
        # Continue with vesicle analysis
        if result is not None and 'features_df' in result:
            vesicle_analysis_results = analyze_vesicle_sizes(pipeline, result['features_df'])
            log_print("Pipeline and vesicle analysis completed successfully!")
        else:
            log_print("Pipeline failed to return usable results.")
    
    log_print(f"--- Pipeline run completed at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Error in pipeline: {str(e)}")
        import traceback
        log_print(traceback.format_exc())
    finally:
        log_file.close() 