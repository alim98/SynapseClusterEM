"""
python synapse_sampling/run_synapse_pipeline_with_sampling.py --use_connectome --policy dummy --batch_size 5 --verbose
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from glob import glob
import time
import datetime


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synapse import config
from synapse_pipeline import SynapsePipeline
from vesicle_size_visualizer import (
    compute_vesicle_cloud_sizes,
    create_umap_with_vesicle_sizes,
    analyze_vesicle_sizes_by_cluster,
    count_bboxes_in_clusters,
    plot_bboxes_in_clusters
)
from newdl.dataloader3 import Synapse3DProcessor

from synapse_sampling.adapter import SynapseConnectomeAdapter, ConnectomeDataset
from synapse_sampling.inference_patch import patch_extract_features, patch_extract_stage_specific_features

log_file = open("pipeline_log.txt", "a")

def log_print(*args, **kwargs):
    """Custom print function that prints to both console and log file"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)
    log_file.flush()  


def configure_pipeline_args():
    """Configure pipeline arguments by extending the existing config object"""
    
    parser = argparse.ArgumentParser(description="Run the synapse analysis pipeline with connectome data")
    
    
    parser.add_argument("--only_vesicle_analysis", action="store_true",
                        help="Only run vesicle size analysis on existing results")
    
    
    parser.add_argument("--extraction_method", type=str, choices=['standard', 'stage_specific'],default='standard',
                       help="Method to extract features ('standard' or 'stage_specific')")
    parser.add_argument("--layer_num", type=int,default=20, 
                       help="Layer number to extract features from when using stage_specific method")
    
    
    parser.add_argument("--pooling_method", type=str, choices=['avg', 'max', 'concat_avg_max', 'spp'],default='avg',
                       help="Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')")
    
    
    parser.add_argument("--use_connectome", action="store_true",
                        help="Use connectome data instead of local files")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of samples to load from connectome")
    parser.add_argument("--policy", type=str, choices=['random', 'dummy'], default='dummy',
                        help="Sampling policy for connectome data")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose information during sampling")
    
    
    config.parse_args()
    
    
    args, _ = parser.parse_known_args()
    
    
    if args.only_vesicle_analysis:
        config.only_vesicle_analysis = True
    else:
        config.only_vesicle_analysis = False
    
    
    if args.extraction_method:
        config.extraction_method = args.extraction_method
    
    if args.layer_num:
        config.layer_num = args.layer_num
    
    
    if args.pooling_method:
        config.pooling_method = args.pooling_method
        
    
    config.use_connectome = args.use_connectome
    config.connectome_batch_size = args.batch_size
    config.connectome_policy = args.policy
    config.connectome_verbose = args.verbose
    
    return config


def run_pipeline_with_connectome(config, timestamp):
    """
    Run the pipeline using connectome data instead of local files.
    
    This is a modified version of the pipeline that uses our adapter
    instead of the original data loading process.
    
    Args:
        config: Configuration object
        timestamp: Timestamp for creating output directories
        
    Returns:
        dict: Pipeline results
    """
    log_print(f"Running pipeline with connectome data sampling...")
    log_print(f"  Batch size: {config.connectome_batch_size}")
    log_print(f"  Policy: {config.connectome_policy}")
    log_print(f"  Verbose: {config.connectome_verbose}")
    
    
    results_base_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)
    
    
    csv_dir = os.path.join(results_base_dir, "csv_outputs")
    clustering_dir = os.path.join(results_base_dir, "clustering_results")
    gifs_dir = os.path.join(results_base_dir, "gifs")
    visualization_dir = os.path.join(results_base_dir, "visualizations")
    features_dir = os.path.join(results_base_dir, "features")
    
    
    for directory in [csv_dir, clustering_dir, gifs_dir, visualization_dir, features_dir]:
        os.makedirs(directory, exist_ok=True)
    
    
    pipeline = SynapsePipeline(config)
    pipeline.results_parent_dir = results_base_dir
    
    
    processor = Synapse3DProcessor(size=config.size)
    processor.normalize_volume = True
    
    
    
    log_print(f"Creating ConnectomeDataset with policy: {config.connectome_policy}")
    dataset = ConnectomeDataset(
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        batch_size=config.connectome_batch_size,
        policy=config.connectome_policy,
        verbose=config.connectome_verbose
    )
    
    
    pipeline.dataset = dataset
    pipeline.vol_data_dict = dataset.vol_data_dict
    pipeline.syn_df = dataset.synapse_df
    
    
    pipeline.load_model()
    
    
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    pooling_method = getattr(config, 'pooling_method', 'avg')
    
    try:
        
        log_print("Extracting features...")
        
        
        if extraction_method == 'stage_specific':
            log_print(f"Using patched stage-specific feature extraction with layer {layer_num}")
            pipeline.features_df = patch_extract_stage_specific_features(
                pipeline.model, 
                pipeline.dataset, 
                config,
                layer_num=layer_num,
                pooling_method=pooling_method
            )
        else:
            log_print("Using patched standard feature extraction")
            pipeline.features_df = patch_extract_features(
                pipeline.model, 
                pipeline.dataset, 
                config,
                pooling_method=pooling_method
            )
            
        
        features_path = os.path.join(features_dir, f"features_{timestamp}.csv")
        pipeline.features_df.to_csv(features_path, index=False)
        log_print(f"Features saved to {features_path}")
        
        
        log_print("Clustering features...")
        pipeline.cluster_features()
        
        
        log_print("Creating visualizations...")
        pipeline.create_dimension_reduction_visualizations()
        
        log_print(f"Pipeline with connectome data completed successfully!")
        log_print(f"Results saved to: {results_base_dir}")
        
        return {
            "features_df": pipeline.features_df,
            "model": pipeline.model,
            "dataset": pipeline.dataset,
            "results_dir": results_base_dir
        }
    except Exception as e:
        log_print(f"Error during pipeline execution: {str(e)}")
        import traceback
        log_print(traceback.format_exc())
        return None


def main():
    """Main function to run the pipeline"""
    log_print(f"\n--- Starting pipeline run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    
    configure_pipeline_args()
    
    
    log_print(f"Running with configuration:")
    log_print(f"  Segmentation Type: {config.segmentation_type}")
    log_print(f"  Alpha: {config.alpha}")
    log_print(f"  Feature Extraction Method: {getattr(config, 'extraction_method', 'standard')}")
    if getattr(config, 'extraction_method', 'standard') == 'stage_specific':
        log_print(f"  Layer Number: {getattr(config, 'layer_num', 20)}")
    log_print(f"  Pooling Method: {getattr(config, 'pooling_method', 'avg')}")
    
    
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    pooling_method = getattr(config, 'pooling_method', 'avg')
    
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    global log_file
    log_file.close()  
    
    
    results_base_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)
    log_file = open(os.path.join(results_base_dir, "pipeline_log.txt"), "w")
    
    
    original_csv_dir = config.csv_output_dir
    original_clustering_dir = config.clustering_output_dir
    original_gifs_dir = config.save_gifs_dir
    
    
    config.csv_output_dir = os.path.join(results_base_dir, "csv_outputs")
    config.clustering_output_dir = os.path.join(results_base_dir, "clustering_results")
    config.save_gifs_dir = os.path.join(results_base_dir, "gifs")
    
    
    config.report_output_dir = os.path.join(results_base_dir, "reports")
    
    log_print(f"Creating parent results directory with timestamp: {timestamp}")
    log_print(f"  Parent directory: {results_base_dir}")
    log_print(f"  CSV output: {config.csv_output_dir}")
    log_print(f"  Clustering output: {config.clustering_output_dir}")
    log_print(f"  GIFs output: {config.save_gifs_dir}")
    log_print(f"  Reports output: {config.report_output_dir}")
    
    if hasattr(config, 'only_vesicle_analysis') and config.only_vesicle_analysis:
        log_print("Running only vesicle analysis on existing results...")
        
        log_print("Vesicle analysis functionality will be implemented soon.")
    else:
        log_print("Running full pipeline...")
        
        
        if hasattr(config, 'use_connectome') and config.use_connectome:
            try:
                result = run_pipeline_with_connectome(config, timestamp)
                if result:
                    log_print("Pipeline with connectome data completed successfully!")
            except Exception as e:
                log_print(f"Error during pipeline execution with connectome data: {str(e)}")
                import traceback
                log_print(traceback.format_exc())
        else:
            
            try:
                log_print("Starting standard pipeline.run_full_pipeline...")
                pipeline = SynapsePipeline(config)
                result = pipeline.run_full_pipeline(
                    seg_type=config.segmentation_type,
                    alpha=config.alpha,
                    extraction_method=extraction_method,
                    layer_num=layer_num,
                    pooling_method=pooling_method
                )
                log_print("Pipeline.run_full_pipeline completed")
            except Exception as e:
                log_print(f"Error during standard pipeline execution: {str(e)}")
                import traceback
                log_print(traceback.format_exc())
    
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