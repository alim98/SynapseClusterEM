import os
import shutil
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from synapse_analysis.data.data_loader import Synapse3DProcessor, load_all_volumes, load_synapse_data
from logger import SynapseLogger

# Initialize the logger properly
logger = SynapseLogger()

def create_sample_visualizations(args, seg_type, seg_output_dir, drive_dir=None):
    """
    Create sample visualizations of synapse data for inspection.
    
    Args:
        args: Command line arguments
        seg_type: Segmentation type being processed
        seg_output_dir: Output directory for the current segmentation type
        drive_dir: Google Drive directory for Colab (optional)
        
    Returns:
        Path to the sample visualization directory
    """
    try:
        sample_vis_dir = Path(seg_output_dir) / "sample_visualizations"
        os.makedirs(sample_vis_dir, exist_ok=True)
        
        # Load first few samples to generate visualizations
        excel_dir = args.excel_dir
        print(f"Excel directory: {excel_dir}")
        print(f"Excel directory exists: {os.path.exists(excel_dir)}")
        
        # Load synapse data - fixed to include bbox_names parameter
        synapse_data = load_synapse_data(args.bbox_names, excel_dir)
        if synapse_data is None or len(synapse_data) == 0:
            logger.error("No synapse data found. Check excel_dir path.")
            return sample_vis_dir
        
        # Print synapse_data type for debugging
        print(f"Synapse data type: {type(synapse_data)}")
        
        # Load volumes - fixed parameter order
        volumes = load_all_volumes(
            args.bbox_names,
            args.raw_base_dir, 
            args.seg_base_dir, 
            args.add_mask_base_dir
        )
        
        if not volumes:
            logger.error("No volumes loaded. Check data paths.")
            return sample_vis_dir
        
        # Create a processor for each volume
        processors = {}
        for bbox_name, volume_data in volumes.items():
            # Unpack the tuple correctly - volume_data is a tuple of (raw_vol, seg_vol, add_mask_vol)
            raw_vol, seg_vol, add_mask_vol = volume_data
            
            # Create a processor with the correct parameters
            # The Synapse3DProcessor doesn't take volume data in its constructor
            processor = Synapse3DProcessor(
                size=tuple(args.size)  # Convert list to tuple
            )
            
            # Store the volume data separately
            processor.raw_vol = raw_vol
            processor.seg_vol = seg_vol
            processor.add_mask_vol = add_mask_vol
            
            processors[bbox_name] = processor
        
        # Generate visualizations for a few samples
        # Handle synapse_data based on its type
        if isinstance(synapse_data, pd.DataFrame):
            # If it's a DataFrame, group by bbox_name
            grouped = synapse_data.groupby('bbox_name')
            num_samples = min(5, len(synapse_data))
            sample_count = 0
            
            for bbox_name, group in grouped:
                if bbox_name not in processors:
                    continue
                    
                processor = processors[bbox_name]
                
                for i, (_, syn_info) in enumerate(group.iterrows()):
                    if sample_count >= num_samples:
                        break
                        
                    # Create visualization
                    try:
                        # Get coordinates
                        x, y, z = int(syn_info.get('x', 0)), int(syn_info.get('y', 0)), int(syn_info.get('z', 0))
                        
                        # Extract a small region around the coordinates if possible
                        raw_vol = processor.raw_vol
                        if raw_vol is not None and raw_vol.size > 0:
                            # Get dimensions
                            depth, height, width = raw_vol.shape
                            
                            # Ensure coordinates are within bounds
                            x = min(max(x, 0), width - 1)
                            y = min(max(y, 0), height - 1)
                            z = min(max(z, 0), depth - 1)
                            
                            # Extract a slice
                            slice_img = raw_vol[z, :, :]
                            
                            # Create visualization
                            plt.figure(figsize=(5, 5))
                            plt.imshow(slice_img, cmap='gray')
                            plt.title(f"Sample {i+1}: {bbox_name}, Synapse {syn_info.get('Var1', i)}")
                            plt.axis('off')
                        else:
                            # Create a placeholder image
                            plt.figure(figsize=(5, 5))
                            plt.text(0.5, 0.5, f"Synapse {syn_info.get('Var1', i)}", 
                                    horizontalalignment='center', verticalalignment='center')
                            plt.axis('off')
                        
                        sample_vis_path = sample_vis_dir / f"sample_{i+1}_{bbox_name}_synapse_{syn_info.get('Var1', i)}.png"
                        plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        sample_count += 1
                    except Exception as e:
                        print(f"Error processing synapse: {e}")
                        continue
        else:
            # Create simple placeholder visualizations
            for i in range(5):
                plt.figure(figsize=(5, 5))
                plt.text(0.5, 0.5, f"Sample {i+1}", 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                
                sample_vis_path = sample_vis_dir / f"sample_{i+1}.png"
                plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Sample visualizations saved to {sample_vis_dir}")
        
        # Copy visualizations to Google Drive if in Colab
        if drive_dir:
            drive_vis_dir = Path(drive_dir) / "sample_visualizations"
            os.makedirs(drive_vis_dir, exist_ok=True)
            for f in sample_vis_dir.glob("*.png"):
                shutil.copy(f, drive_vis_dir / f.name)
        
        logger.info(f"Sample visualizations created at {drive_dir / 'sample_visualizations' if drive_dir else None}")
        return sample_vis_dir
    except Exception as e:
        logger.warning(f"Error creating sample visualizations: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# Main execution block
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("./outputs/test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize args with required parameters
    args = argparse.Namespace(
        raw_base_dir="data/7_bboxes_plus_seg/raw",
        seg_base_dir="data/7_bboxes_plus_seg/seg",
        add_mask_base_dir="data/vesicle_cloud__syn_interface__mitochondria_annotation",
        excel_dir="data/7_bboxes_plus_seg",  # Contains bbox1.xlsx through bbox7.xlsx
        output_dir=str(output_dir),
        checkpoint_path="hemibrain_production.checkpoint",
        bbox_names=["bbox1", "bbox2"],  # Uncomment others if needed
        # bbox_names=["bbox3", "bbox4", "bbox5", "bbox6", "bbox7"],
        size=[80, 80],  # [height, width]
        num_frames=16  # Default value for 3D volumes
    )
    
    # Call the function with initialized arguments
    create_sample_visualizations(
        args=args,
        seg_type=1,  # Default segmentation type
        seg_output_dir=output_dir,
        drive_dir=None  # No Google Drive by default
    )