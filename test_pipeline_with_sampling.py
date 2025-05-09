"""
Test script for running the synapse pipeline with connectome sampling
"""

import os
import sys
import datetime
from run_synapse_pipeline_with_sampling import configure_pipeline_args, run_pipeline_with_connectome

def main():
    """Run the pipeline with connectome sampling"""
    # Configure arguments
    sys.argv = [
        "test_pipeline_with_sampling.py",
        "--use_connectome",
        "--batch_size", "2",
        "--policy", "dummy",
        "--verbose",
        "--extraction_method", "stage_specific",
        "--layer_num", "10",
        "--pooling_method", "avg"
    ]
    
    # Configure pipeline
    config = configure_pipeline_args()
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Run pipeline
    print(f"Running pipeline with timestamp: {timestamp}")
    
    # Create results directory
    results_base_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Create subdirectories
    csv_dir = os.path.join(results_base_dir, "csv_outputs")
    features_dir = os.path.join(results_base_dir, "features")
    
    # Create directories
    for directory in [csv_dir, features_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Import necessary modules
    from synapse_pipeline import SynapsePipeline
    from newdl.dataloader3 import Synapse3DProcessor
    from synapse_sampling.adapter import ConnectomeDataset
    from inference_patch import patch_extract_features, patch_extract_stage_specific_features
    
    # Initialize pipeline
    pipeline = SynapsePipeline(config)
    pipeline.results_parent_dir = results_base_dir
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    processor.normalize_volume = True
    
    # Create connectome dataset
    print(f"Creating ConnectomeDataset with policy: {config.connectome_policy}")
    dataset = ConnectomeDataset(
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        batch_size=config.connectome_batch_size,
        policy=config.connectome_policy,
        verbose=config.connectome_verbose
    )
    
    # Store dataset and vol_data_dict in pipeline
    pipeline.dataset = dataset
    pipeline.vol_data_dict = dataset.vol_data_dict
    pipeline.syn_df = dataset.synapse_df
    
    # Load model
    print("Loading model...")
    pipeline.load_model()
    
    try:
        # Test both extraction methods
        extraction_method = getattr(config, 'extraction_method', 'standard')
        layer_num = getattr(config, 'layer_num', 10)
        pooling_method = getattr(config, 'pooling_method', 'avg')
        
        if extraction_method == 'stage_specific':
            print(f"Testing stage-specific feature extraction with layer {layer_num}")
            features_df = patch_extract_stage_specific_features(
                pipeline.model, 
                pipeline.dataset, 
                config,
                layer_num=layer_num,
                pooling_method=pooling_method
            )
            
            # Save features
            features_path = os.path.join(features_dir, f"stage_specific_features_layer{layer_num}_{timestamp}.csv")
            features_df.to_csv(features_path, index=False)
            print(f"Stage-specific features saved to {features_path}")
        else:
            print("Testing standard feature extraction")
            features_df = patch_extract_features(
                pipeline.model, 
                pipeline.dataset, 
                config,
                pooling_method=pooling_method
            )
            
            # Save features
            features_path = os.path.join(features_dir, f"standard_features_{timestamp}.csv")
            features_df.to_csv(features_path, index=False)
            print(f"Standard features saved to {features_path}")
        
        print("Test completed successfully!")
        print(f"Results saved to: {results_base_dir}")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 