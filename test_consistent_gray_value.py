import os
import sys
import numpy as np
import torch
import pandas as pd
import imageio
import io
import base64
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import necessary modules
try:
    from synapse.utils.config import config
    from synapse.gif_umap.GifUmap import create_gif_from_volume
    from newdl.dataset3 import SynapseDataset
    from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def initialize_dataset_with_no_normalization():
    """
    Initialize a SynapseDataset with normalization disabled.
    """
    try:
        print("Initializing dataset with normalization disabled...")
        
        # Initialize data loader
        data_loader = SynapseDataLoader(
            raw_base_dir=config.raw_base_dir,
            seg_base_dir=config.seg_base_dir,
            add_mask_base_dir=config.add_mask_base_dir
        )
        
        # Load volumes
        vol_data_dict = {}
        for bbox_name in config.bbox_name:
            print(f"Loading volumes for {bbox_name}...")
            raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
            if raw_vol is not None:
                vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
        
        # Check if we have any volumes
        if not vol_data_dict:
            print("No volumes could be loaded. Check the configuration paths.")
            return None
            
        # Load synapse data
        syn_df = pd.DataFrame()
        if config.excel_file:
            try:
                syn_df = pd.concat([
                    pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
                    for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
                ])
                print(f"Loaded synapse data: {len(syn_df)} rows")
            except Exception as e:
                print(f"Error loading Excel files: {e}")
                
        # Initialize processor
        processor = Synapse3DProcessor(size=config.size)
        processor.normalize_volume = False  # Disable volume normalization
        
        # Create dataset
        dataset = SynapseDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=syn_df,
            processor=processor,
            segmentation_type=config.segmentation_type,
            subvol_size=config.subvol_size,
            num_frames=config.num_frames,
            alpha=config.alpha,
            normalize_across_volume=False  # Disable volume normalization
        )
        
        print(f"Successfully created dataset with {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_modified_gif_from_volume(volume, output_path, fps=10, apply_normalization=False):
    """
    Create a GIF from a volume with optional normalization control.
    
    Args:
        volume: 3D array representing volume data
        output_path: Path to save the GIF
        fps: Frames per second
        apply_normalization: Whether to apply normalization
        
    Returns:
        output_path: Path to created GIF
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().detach().numpy()
    
    # Ensure volume is a numpy array
    if not isinstance(volume, np.ndarray):
        raise TypeError(f"Volume must be a numpy array or PyTorch tensor, got {type(volume)}")
    
    # If volume has more than 3 dimensions, squeeze it
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    
    # Prepare frames for GIF
    frames = []
    
    if apply_normalization:
        # Calculate global min/max for consistent scaling
        vol_min, vol_max = volume.min(), volume.max()
        scale_factor = vol_max - vol_min
        
        if scale_factor > 0:  # Avoid division by zero
            for i in range(volume.shape[0]):
                frame = volume[i]
                # Normalize using global min/max
                normalized = (frame - vol_min) / scale_factor
                frames.append((normalized * 255).astype(np.uint8))
        else:
            # If all values are the same, create blank frames
            for i in range(volume.shape[0]):
                frames.append(np.zeros_like(volume[i], dtype=np.uint8))
    else:
        # Use absolute fixed scaling to match dataloader3.py behavior
        # This ensures completely consistent gray values across all samples
        
        # Define same fixed values as in dataloader3.py
        fixed_min = 0.0
        fixed_max = 255.0
        
        # If values are in 0-1 range, scale to 0-255 for processing
        if volume.max() <= 1.0:
            volume = volume * 255.0
            
        for i in range(volume.shape[0]):
            frame = volume[i]
            # Clip to fixed range without any normalization
            clipped = np.clip(frame, fixed_min, fixed_max)
            # Convert to uint8 for GIF
            scaled = clipped.astype(np.uint8)
            frames.append(scaled)
            
        print(f"Using ABSOLUTE fixed gray values: min={fixed_min}, max={fixed_max}")
        print(f"Volume range before clipping: {volume.min():.4f}-{volume.max():.4f}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    
    print(f"Created GIF at {output_path}")
    return output_path

def test_gif_creation_with_real_samples():
    """Test creating GIFs with real samples from the dataset."""
    print("\n--- Testing GIF Creation with Real Samples ---")
    
    # Create output directory
    output_dir = Path("test_output_real")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize dataset with normalization disabled
    dataset = initialize_dataset_with_no_normalization()
    
    if dataset is None or len(dataset) == 0:
        print("Error: Could not initialize dataset or dataset is empty")
        return
    
    # Select a few samples from the dataset
    num_samples = min(20, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    print(f"Processing {num_samples} samples from the dataset")
    
    for i, idx in enumerate(indices):
        print(f"\nProcessing sample {i+1}/{num_samples} (dataset index: {idx})")
        
        # Get the sample from the dataset
        try:
            pixel_values, syn_info, bbox_name = dataset[idx]
            
            # Print some info about the sample
            print(f"Sample info - bbox: {bbox_name}")
            print(f"Pixel values shape: {pixel_values.shape}")
            print(f"Value range: min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}")
            
            # Create two versions of the GIF - with and without normalization
            # Without normalization
            output_path_no_norm = output_dir / f"sample_{i}_no_norm.gif"
            create_modified_gif_from_volume(
                pixel_values, 
                str(output_path_no_norm), 
                fps=5, 
                apply_normalization=False
            )
            
            # # With normalization
            # output_path_with_norm = output_dir / f"sample_{i}_with_norm.gif"
            # create_modified_gif_from_volume(
            #     pixel_values, 
            #     str(output_path_with_norm), 
            #     fps=5, 
            #     apply_normalization=True
            # )
            
            # Print file sizes
            if os.path.exists(output_path_no_norm):
                size_no_norm = os.path.getsize(output_path_no_norm)
                print(f"GIF without normalization - Size: {size_no_norm} bytes")
            
            # if os.path.exists(output_path_with_norm):
            #     size_with_norm = os.path.getsize(output_path_with_norm)
            #     print(f"GIF with normalization - Size: {size_with_norm} bytes")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing GIF creation with real samples")
    test_gif_creation_with_real_samples()
    print("\nTest completed")

"""
CONSISTENT GRAY VALUE GUIDE

This guide explains how to maintain consistent gray values across the entire pipeline
from dataset loading to GIF creation.

KEY SETTINGS TO MAINTAIN CONSISTENT GRAY VALUES:

1. During Dataset Initialization:
   - Set processor.normalize_volume = False
   - Set normalize_across_volume = False
   Example:
     ```
     processor = Synapse3DProcessor(size=config.size)
     processor.normalize_volume = False
     
     dataset = SynapseDataset(
         vol_data_dict=vol_data_dict,
         synapse_df=syn_df,
         processor=processor,
         segmentation_type=config.segmentation_type,
         subvol_size=config.subvol_size,
         num_frames=config.num_frames,
         alpha=config.alpha,
         normalize_across_volume=False  # This is crucial!
     )
     ```

2. During GIF Creation:
   - Use create_gif_from_volume WITHOUT normalization
   - The function in GifUmap.py has been modified to NEVER apply normalization
   Example:
     ```
     # The create_gif_from_volume function has been modified to:
     # - Never apply normalization
     # - Preserve the original consistent gray values
     gif_path, frames = create_gif_from_volume(volume, str(output_path), fps=5)
     ```

3. How to test if gray values are consistent:
   Run the test_gif_creation.py script, which will:
   - Initialize the dataset with normalization disabled
   - Create GIFs that preserve the original gray values
   - Output GIFs to the test_output_real directory

TESTING:
To test whether the gray values are consistent:
1. Create GIFs using the code examples above
2. Visually inspect the GIFs:
   - Consistent gray values: The background intensity should look similar across all frames
   - Inconsistent gray values: The background may appear brighter in some frames, darker in others

If your GIFs have inconsistent gray values, check:
1. That you're using the modified create_gif_from_volume function without normalization
2. That the dataset is initialized with normalization disabled
3. That the SynapseDataLoader is configured correctly (check initialize_dataset_from_newdl function)

CODE LOCATIONS THAT HAVE BEEN FIXED:

1. In synapse/gif_umap/GifUmap.py:
   - The create_gif_from_volume function has been modified to never apply normalization
   - The initialize_dataset_from_newdl function sets processor.normalize_volume = False and 
     normalize_across_volume = False

2. In test_gif_creation.py:
   - The create_modified_gif_from_volume function allows controlling normalization
   - You can use apply_normalization=False to preserve consistent gray values

For consistent visualization, ensure all places that output images (including slice views)
use consistent normalization approaches. For debugging, you can generate both normalized
and non-normalized versions to compare the difference.
""" 