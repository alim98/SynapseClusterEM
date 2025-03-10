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
    
    # Let config parse its arguments first
    config.parse_args()
    
    # Parse our additional arguments
    args, _ = parser.parse_known_args()
    
    # Add our arguments to config
    if args.only_vesicle_analysis:
        config.only_vesicle_analysis = True
    else:
        config.only_vesicle_analysis = False
    
    return config


def main():
    """Main function to run the pipeline"""
    log_print(f"\n--- Starting pipeline run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # Configure pipeline arguments
    configure_pipeline_args()
    
    # Log configuration details
    log_print(f"Using config with:")
    log_print(f"  seg_type: {config.segmentation_type}")
    log_print(f"  alpha: {config.alpha}")
    log_print(f"  csv_output_dir: {config.csv_output_dir}")
    log_print(f"  skip_feature_extraction: {config.skip_feature_extraction if hasattr(config, 'skip_feature_extraction') else False}")
    log_print(f"  skip_clustering: {config.skip_clustering if hasattr(config, 'skip_clustering') else False}")
    log_print(f"  only_vesicle_analysis: {config.only_vesicle_analysis if hasattr(config, 'only_vesicle_analysis') else False}")
    
    # Set up output directories
    base_output_dir = config.output_dir if hasattr(config, 'output_dir') else "results"
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    log_print(f"Run ID: {timestamp}")
    
    # Create a parent directory for this specific run based on config and timestamp
    run_name = f"{timestamp}_seg{config.segmentation_type}_alpha{config.alpha}"
    run_dir = os.path.join(base_output_dir, run_name)
    log_print(f"Output directory: {run_dir}")
    
    # Create subdirectories within the run directory
    feature_output_dir = os.path.join(run_dir, "features")
    clustering_output_dir = os.path.join(run_dir, "clustering_results")
    viz_output_dir = os.path.join(run_dir, "visualizations")
    sample_viz_output_dir = os.path.join(run_dir, "sample_visualizations") 
    presynapse_output_dir = os.path.join(run_dir, "presynapse_analysis")
    vesicle_output_dir = os.path.join(run_dir, "vesicle_analysis")
    csv_output_dir = os.path.join(run_dir, "csv_outputs")
    
    # Also update the config's csv_output_dir to the new location
    config.csv_output_dir = csv_output_dir
    
    # Create directories
    for directory in [base_output_dir, run_dir, feature_output_dir, clustering_output_dir, 
                    viz_output_dir, sample_viz_output_dir, presynapse_output_dir,
                    vesicle_output_dir, csv_output_dir]:
        os.makedirs(directory, exist_ok=True)
        log_print(f"Created directory: {directory}")
    
    # Initialize the pipeline
    pipeline = SynapsePipeline(config)
    
    # If we're only doing vesicle analysis on existing results
    if hasattr(config, 'only_vesicle_analysis') and config.only_vesicle_analysis:
        log_print("Running only vesicle size analysis on existing results...")
        
        # Generate timestamp for this run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create a parent directory for this specific run based on config and timestamp
        run_name = f"{timestamp}_vesicle_analysis_seg{config.segmentation_type}_alpha{config.alpha}"
        run_dir = os.path.join(base_output_dir, run_name)
        
        # Create subdirectories within the run directory
        csv_output_dir = os.path.join(run_dir, "csv_outputs")
        vesicle_output_dir = os.path.join(run_dir, "vesicle_analysis")
        
        # Create necessary directories
        for directory in [base_output_dir, run_dir, csv_output_dir, vesicle_output_dir]:
            os.makedirs(directory, exist_ok=True)
            log_print(f"Created directory: {directory}")
        
        # Update the config's csv_output_dir to the new location
        config.csv_output_dir = csv_output_dir
        
        # Look for feature files in various directory structures
        feature_paths = []
        
        # Check in run-specific directory (current format)
        feature_paths.append(os.path.join(run_dir, "clustering_results", "clustered_features.csv"))
        feature_paths.append(os.path.join(run_dir, "features", "features.csv"))
        
        # Check in legacy named directories
        feature_paths.append(os.path.join(base_output_dir, f"clustering_results_seg{config.segmentation_type}_alpha{config.alpha}", "clustered_features.csv"))
        feature_paths.append(os.path.join(base_output_dir, f"features_seg{config.segmentation_type}_alpha{config.alpha}", "features.csv"))
        
        # Check in timestamp directories (without vesicle_analysis prefix) - match any timestamp
        timestamp_dirs = glob(os.path.join(base_output_dir, f"*_seg{config.segmentation_type}_alpha{config.alpha}"))
        for tdir in timestamp_dirs:
            feature_paths.append(os.path.join(tdir, "clustering_results", "clustered_features.csv"))
            feature_paths.append(os.path.join(tdir, "features", "features.csv"))
        
        # Check in csv_outputs directories (various formats)
        feature_paths.append(os.path.join(csv_output_dir, f"features_seg{config.segmentation_type}_alpha{config.alpha}_0.csv"))
        feature_paths.append(os.path.join(base_output_dir, "csv_outputs", f"features_seg{config.segmentation_type}_alpha{config.alpha}_0.csv"))
        feature_paths.append(f"csv_outputs/features_seg{config.segmentation_type}_alpha{config.alpha}_0.csv")
        
        # Legacy exact paths
        feature_paths.append("csv_outputs\\features_seg1_alpha1.0\\features_seg1_alpha1_0.csv")
        
        log_print("Searching for feature files in these locations:")
        for path in feature_paths:
            log_print(f"  - {path} (exists: {os.path.exists(path)})")
        
        features_df = None
        for path in feature_paths:
            if os.path.exists(path):
                log_print(f"Found features at {path}")
                features_df = pd.read_csv(path)
                log_print(f"Loaded {len(features_df)} rows of feature data")
                log_print(f"Feature columns: {features_df.columns.tolist()}")
                break
                
        if features_df is None:
            log_print("Error: Could not find features or clustered features in any of these locations:")
            for path in feature_paths:
                log_print(f"  - {path}")
            return
        
        # Check if the features have cluster assignments already
        if 'cluster' not in features_df.columns:
            log_print("Features don't have cluster assignments. Running clustering...")
            # Initialize the pipeline
            pipeline = SynapsePipeline(config)
            pipeline.features_df = features_df
            pipeline.cluster_features(clustering_output_dir)
            features_df = pipeline.features_df
            
        # Load vesicle sizes or compute them if needed
        vesicle_csvs = []
        
        # Check both old and new directory structures
        vesicle_paths = [
            os.path.join(csv_output_dir, "bbox*.csv"),
            os.path.join(base_output_dir, "csv_outputs", "bbox*.csv"),
            "csv_outputs/bbox*.csv"
        ]
        
        for path in vesicle_paths:
            found_csvs = glob(path)
            if found_csvs:
                vesicle_csvs.extend(found_csvs)
                log_print(f"Found vesicle CSVs at {path}: {len(found_csvs)} files")
        
        if not vesicle_csvs:
            log_print("Loading data for vesicle size calculation...")
            pipeline = SynapsePipeline(config)
            pipeline.load_data()
            vesicle_df = compute_vesicle_cloud_sizes(
                pipeline.syn_df, pipeline.vol_data_dict, config, 
                csv_output_dir  # Use the new CSV output directory
            )
        else:
            vesicle_df = pd.concat([pd.read_csv(f) for f in vesicle_csvs])
            log_print(f"Loaded {len(vesicle_df)} rows of vesicle data")
        
        # Create UMAP with vesicle sizes
        log_print("Creating UMAP with vesicle sizes...")
        merged_df, _, _ = create_umap_with_vesicle_sizes(
            features_df, vesicle_df, vesicle_output_dir
        )
        
        # Analyze vesicle sizes by cluster
        log_print("Analyzing vesicle sizes by cluster...")
        analyze_vesicle_sizes_by_cluster(merged_df, vesicle_output_dir)
        
        # Analyze bounding boxes in clusters
        log_print("Analyzing bounding box distribution in clusters...")
        cluster_counts = count_bboxes_in_clusters(features_df)
        plot_bboxes_in_clusters(cluster_counts, vesicle_output_dir)
        
        log_print(f"Vesicle analysis completed. Results saved to {vesicle_output_dir}")
        return
        
    # Run the full pipeline with stages as specified
    log_print(f"Running synapse analysis pipeline with seg_type={config.segmentation_type}, alpha={config.alpha}")
    
    # Load data and model
    log_print("Starting data loading...")
    pipeline.load_data()
    log_print("Starting model loading...")
    pipeline.load_model()
    
    # Extract features if not skipped
    skip_feature_extraction = hasattr(config, 'skip_feature_extraction') and config.skip_feature_extraction
    if not skip_feature_extraction:
        log_print("Starting feature extraction...")
        pipeline.extract_features(config.segmentation_type, config.alpha)
    else:
        # Load features from existing file
        features_path = os.path.join(feature_output_dir, "features.csv")
        log_print(f"Attempting to load features from {features_path}")
        if os.path.exists(features_path):
            pipeline.features_df = pd.read_csv(features_path)
            log_print(f"Loaded {len(pipeline.features_df)} rows of feature data from {features_path}")
        else:
            log_print(f"Warning: Could not find features at {features_path}, extracting features anyway")
            pipeline.extract_features(config.segmentation_type, config.alpha)
    
    # Cluster features if not skipped
    skip_clustering = hasattr(config, 'skip_clustering') and config.skip_clustering
    if not skip_clustering:
        log_print("Starting clustering...")
        pipeline.cluster_features(clustering_output_dir)
    else:
        # Load clustered features from existing file
        clustered_features_path = os.path.join(clustering_output_dir, "clustered_features.csv")
        log_print(f"Attempting to load clustered features from {clustered_features_path}")
        if os.path.exists(clustered_features_path):
            pipeline.features_df = pd.read_csv(clustered_features_path)
            log_print(f"Loaded {len(pipeline.features_df)} rows of clustered feature data")
        else:
            log_print(f"Warning: Could not find clustered features at {clustered_features_path}, clustering anyway")
            pipeline.cluster_features(clustering_output_dir)
    
    # Create visualizations if not skipped
    skip_visualization = hasattr(config, 'skip_visualization') and config.skip_visualization
    if not skip_visualization:
        log_print("Starting visualizations...")
        pipeline.create_dimension_reduction_visualizations(viz_output_dir)
        pipeline.create_cluster_sample_visualizations(
            config.num_samples if hasattr(config, 'num_samples') else 4, 
            config.attention_layer if hasattr(config, 'attention_layer') else 20, 
            sample_viz_output_dir
        )
    
    # Run presynapse analysis if not skipped
    skip_presynapse_analysis = hasattr(config, 'skip_presynapse_analysis') and config.skip_presynapse_analysis
    if not skip_presynapse_analysis:
        log_print("Starting presynapse analysis...")
        pipeline.run_presynapse_analysis(presynapse_output_dir)
    
    # Compute and analyze vesicle sizes
    log_print("Starting vesicle size analysis...")
    vesicle_df = compute_vesicle_cloud_sizes(
        pipeline.syn_df, pipeline.vol_data_dict, config, 
        csv_output_dir  # Use the new CSV output directory
    )
    
    merged_df, _, _ = create_umap_with_vesicle_sizes(
        pipeline.features_df, vesicle_df, vesicle_output_dir
    )
    
    analyze_vesicle_sizes_by_cluster(merged_df, vesicle_output_dir)
    
    # Analyze bounding boxes in clusters
    log_print("Analyzing bounding box distribution in clusters...")
    cluster_counts = count_bboxes_in_clusters(pipeline.features_df)
    plot_bboxes_in_clusters(cluster_counts, vesicle_output_dir)
    
    log_print("Pipeline completed successfully!")
    log_print(f"Results saved to {run_dir}")  # Update to show the run directory
    log_print(f"--- Pipeline run completed at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Error in pipeline: {str(e)}")
        import traceback
        log_print(traceback.format_exc())
    finally:
        log_file.close() 