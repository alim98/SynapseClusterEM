import os
import logging
import sys

# Add the parent directory to the path so we can import from sample_fig_compare_crop
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newdl.sample_fig_compare_crop import visualize_comparison
from synapse.utils.config import config
from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_normalized_crops")

def main():
    # Parse configuration
    config.parse_args()
    
    # Specific parameters for the crops we need
    bbox_name = "bbox3"
    var1 = "non_spine_synapse_009"
    weights = [0.3, 0.7]  # The missing weights
    
    # Set segmentation type
    segmentation_type = 5
    
    # Output directory
    save_gifs_dir = "newdl/crop_comparison"
    logger.info(f"Output directory: {save_gifs_dir}")
    
    # Initialize data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    logger.info(f"Loading volumes for {bbox_name}...")
    raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
    if raw_vol is None:
        logger.error(f"Could not load volumes for {bbox_name}")
        return
    
    vol_data_dict = {bbox_name: (raw_vol, seg_vol, add_mask_vol)}
    
    # Load synapse data
    excel_path = os.path.join(config.excel_file, f"{bbox_name}.xlsx")
    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found: {excel_path}")
        return
    
    logger.info(f"Loading Excel file: {excel_path}")
    bbox_df = pd.read_excel(excel_path).assign(bbox_name=bbox_name)
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    
    # Process each weight
    for weight in weights:
        logger.info(f"Generating normalized intelligent crop with weight {weight}...")
        
        # Generate the normalized intelligent crop
        try:
            visualize_comparison(
                syn_df=bbox_df,
                bbox_name=bbox_name,
                var1=var1,
                vol_data_dict=vol_data_dict,
                save_gifs_dir=save_gifs_dir,
                segmentation_type=segmentation_type,
                processor=processor,
                subvol_size=config.subvol_size,
                num_frames=config.num_frames,
                alpha=config.alpha,
                presynapse_weight=weight,
                normalize_presynapse_size=True,
                target_percentage=0.15,
                size_tolerance=0.1
            )
            
            # Check if the file was created
            output_path = os.path.join(save_gifs_dir, f"{bbox_name}_{var1}_intelligent_w{weight}_normalized.gif")
            if os.path.exists(output_path):
                logger.info(f"Successfully created normalized intelligent crop with weight {weight}: {output_path}")
            else:
                logger.error(f"Failed to create normalized intelligent crop with weight {weight}")
                
        except Exception as e:
            logger.error(f"Error generating normalized intelligent crop with weight {weight}: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 