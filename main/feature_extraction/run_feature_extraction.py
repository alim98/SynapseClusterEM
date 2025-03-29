"""
Run feature extraction on multiple segmentation types

This script runs feature extraction for segmentation types 11 and 13,
saving the results in type-specific folders.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
print(f"Added project root to path: {project_root}")

try:
    from synapse import config
    from main.feature_extraction.FeatureExtraction import FeatureExtraction
    print("Successfully imported required modules")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Python version: {sys.version}")
    traceback.print_exc()
    sys.exit(1)

def run_feature_extraction(seg_types=[10], alpha=1.0):
    """
    Run feature extraction for multiple segmentation types
    
    Args:
        seg_types: List of segmentation types to process
        alpha: Alpha value for feature extraction
    """
    # Common configuration
    extraction_method = 'stage_specific'
    layer_num = 20  # Using stage_specific layer 20 as requested
    
    # Print current config values for debugging
    print(f"\nCurrent configuration:")
    print(f"- Raw data dir: {getattr(config, 'raw_base_dir', 'Not set')}")
    print(f"- Seg data dir: {getattr(config, 'seg_base_dir', 'Not set')}")
    print(f"- Mask data dir: {getattr(config, 'add_mask_base_dir', 'Not set')}")
    print(f"- Extraction method: {extraction_method}")
    print(f"- Layer number: {layer_num}")
    print(f"- Alpha value: {alpha}")
    
    # Create a master results directory with timestamp
    results_dir = getattr(config, 'results_dir', os.path.join(os.path.dirname(__file__), "results"))
    master_output_dir = Path(results_dir) / f"feature_extraction_{time.strftime('%Y%m%d_%H%M%S')}"
    master_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Master results directory: {master_output_dir}")
    
    # Process each segmentation type
    for seg_type in seg_types:
        print(f"\n{'='*80}")
        print(f"Processing segmentation type {seg_type}")
        print(f"{'='*80}\n")
        
        # Update config for this run
        config.segmentation_type = seg_type
        config.alpha = alpha
        config.extraction_method = extraction_method
        config.layer_num = layer_num
        config.preprocessing = 'intelligent_cropping'
        config.preprocessing_weights = 0.7
        # Create type-specific output directory
        type_output_dir = master_output_dir / f"seg_type_{seg_type}"
        type_output_dir.mkdir(exist_ok=True)
        config.results_dir = str(type_output_dir)
        
        # Create and run feature extraction
        try:
            extractor = FeatureExtraction(config)
            extractor.create_results_directory()
            extractor.load_data()
            extractor.load_model()
            features_df = extractor.extract_features(
                seg_type=seg_type,
                alpha=alpha,
                extraction_method=extraction_method,
                layer_num=layer_num
            )
            
            print(f"Features for segmentation type {seg_type} extracted and saved to: {extractor.results_parent_dir}")
        except Exception as e:
            print(f"Error processing segmentation type {seg_type}: {str(e)}")
            traceback.print_exc()
            print("Continuing with next segmentation type...")
    
    print(f"\nFeature extraction complete. Results saved in {master_output_dir}")

if __name__ == "__main__":
    # Configure paths if needed
    if not hasattr(config, 'results_dir') or not config.results_dir:
        workspace_dir = os.path.abspath(os.path.dirname(__file__))
        config.results_dir = os.path.join(workspace_dir, "results")
    
    try:
        # Run feature extraction for segmentation types 11 and 13
        run_feature_extraction(seg_types=[10])
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 