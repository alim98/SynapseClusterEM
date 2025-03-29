"""
Attention Map GIF Visualization for Multiple Segmentation Types

This script generates animated GIFs of Class Activation Maps (CAM) for segmentation types 10, 11, 12, and 13,
visualizing layer 20 attention for 4 samples of each type.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import io
from PIL import Image

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
print(f"Added project root to path: {project_root}")

# Import from newdl module instead of synapse
from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset2 import SynapseDataset
from synapse import config
from inference import VGG3D, load_and_prepare_data
from multi_layer_cam import SimpleGradCAM, process_single_sample  # Import the necessary functions

def create_attention_gif(result, output_path, fps=10, dpi=100):
    """
    Create an animated GIF of original images and attention maps.
    
    Args:
        result: Dictionary containing original images and layer results
        output_path: Path to save the GIF
        fps: Frames per second for the GIF
        dpi: Resolution of the frames
    
    Returns:
        Path to the saved GIF file
    """
    # Extract data from result
    original_img = result['original_img'][0, 0]  # Shape: [D, H, W]
    n_frames = original_img.shape[0]
    layer_results = result['layer_results']
    bbox_name = result['bbox_name']
    syn_info = result['syn_info']
    
    # Prepare directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process each layer
    for layer_name, cam in layer_results.items():
        # Create output path with layer name
        layer_output_path = output_path.replace('.gif', f'_{layer_name}.gif')
        
        # Create frames for GIF
        frames = []
        
        # Loop through all slices/frames
        for frame_idx in range(n_frames):
            # Create figure with two subplots side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot original image
            axs[0].imshow(original_img[frame_idx], cmap='gray', vmin=0, vmax=1)
            axs[0].set_title('Original', fontsize=10)
            axs[0].axis('off')
            
            # Plot original with attention overlay
            axs[1].imshow(original_img[frame_idx], cmap='gray', vmin=0, vmax=1)
            cam_frame = cam[frame_idx]
            im = axs[1].imshow(cam_frame, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axs[1].set_title(f'Attention Map ({layer_name})', fontsize=10)
            axs[1].axis('off')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
            cbar.set_label('Attention', fontsize=8)
            
            # Add frame number and information
            synapse_id = syn_info.get('Var1', 'Unknown')
            plt.suptitle(f"Sample: {synapse_id} (BBox: {bbox_name}) - Frame {frame_idx+1}/{n_frames}\nSegmentation Type: {config.segmentation_type}, Alpha: {config.alpha}", fontsize=10)
            
            plt.tight_layout()
            
            # Use a safer method to convert the figure to an image array
            # Save figure to a temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            
            # Read the image from buffer
            img = Image.open(buf)
            frame = np.array(img)
            
            frames.append(frame)
            plt.close()
            buf.close()
        
        # Save as GIF
        imageio.mimsave(layer_output_path, frames, fps=fps)
        print(f"Saved attention GIF to: {layer_output_path}")
    
    return layer_output_path

def visualize_samples_attention_as_gifs(model, dataset, sample_indices, output_dir, layers, layer_names):
    """
    Visualizes attention maps for samples as animated GIFs.
    
    Args:
        model: The PyTorch model
        dataset: The dataset containing the samples
        sample_indices: List of sample indices to visualize
        output_dir: Directory to save visualizations
        layers: List of layer identifiers to visualize
        layer_names: List of layer names corresponding to the layers
        
    Returns:
        List of paths to saved GIFs
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_files = []
    
    # Process each sample
    for sample_idx in sample_indices:
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} exceeds dataset size ({len(dataset)}). Skipping.")
            continue
        
        try:
            # Process this sample to get attention maps
            result = process_single_sample(model, dataset, sample_idx, device, layers, layer_names)
            
            # Create GIF for this sample
            bbox_name = result['bbox_name']
            synapse_id = result['syn_info'].get('Var1', f'Sample_{sample_idx}')
            
            # Create output path
            alpha_value = config.alpha if hasattr(config, 'alpha') else 'unknown'
            seg_type = config.segmentation_type if hasattr(config, 'segmentation_type') else 'unknown'
            output_file = os.path.join(output_dir, f"attention_sample_{sample_idx}_bbox_{bbox_name}_alpha{alpha_value}_seg{seg_type}.gif")
            
            # Generate the GIF
            gif_path = create_attention_gif(result, output_file)
            print(f"Attention GIF for sample {sample_idx} created at {gif_path}")
            output_files.append(gif_path)
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return output_files

def run_cam_visualization(seg_types=[10, 11, 12, 13], n_samples=4):
    """
    Generate and visualize attention maps as GIFs for multiple segmentation types.
    
    Args:
        seg_types: List of segmentation types to process
        n_samples: Number of samples to visualize for each segmentation type
    """
    # Set alpha value for visualization
    alpha = 1.0  # Using 1.0 to see clearer differentiation
    
    # Set up model
    print("Initializing model...")
    model = VGG3D()
    model.eval()
    
    # Define layers to analyze - we're focusing on layer 20 as requested
    layers = ["features.20"]
    layer_names = ["Layer_20"]
    
    # Create a master results directory with timestamp
    results_dir = getattr(config, 'results_dir', os.path.join(os.path.dirname(__file__), "results"))
    master_output_dir = Path(results_dir) / f"cam_gif_visualizations_{time.strftime('%Y%m%d_%H%M%S')}"
    master_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Master results directory: {master_output_dir}")
    
    # Load data once - will be reused for all segmentation types
    print("Loading data...")
    vol_data_dict, syn_df = load_and_prepare_data(config)
    processor = Synapse3DProcessor(size=config.size)
    processor.normalize_volume = True  # Ensure volume-wide normalization
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Process each segmentation type
    for seg_type in seg_types:
        print(f"\n{'='*80}")
        print(f"Processing segmentation type {seg_type}")
        print(f"{'='*80}\n")
        
        # Update config for this run
        config.segmentation_type = seg_type
        config.alpha = alpha
        
        # Create dataset with current segmentation type
        dataset = SynapseDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=syn_df,
            processor=processor,
            segmentation_type=seg_type,
            alpha=alpha,
            normalize_across_volume=True
        )
        print(f"Created dataset with segmentation_type={seg_type}, alpha={alpha}")
        print(f"Dataset size: {len(dataset)} samples")
        
        # Create type-specific output directory
        type_output_dir = master_output_dir / f"seg_type_{seg_type}"
        type_output_dir.mkdir(exist_ok=True)
        
        # Select samples for this segmentation type
        # We'll use a deterministic random selection based on seg_type to ensure reproducibility
        random.seed(42 + seg_type)  # Different seed for each segmentation type
        sample_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        print(f"Selected samples for segmentation type {seg_type}: {sample_indices}")
        
        # Generate and visualize attention maps as GIFs
        try:
            output_files = visualize_samples_attention_as_gifs(
                model=model,
                dataset=dataset,
                sample_indices=sample_indices,
                output_dir=type_output_dir,
                layers=layers,
                layer_names=layer_names
            )
            
            print(f"GIF visualizations for segmentation type {seg_type} saved to: {type_output_dir}")
            print(f"Generated {len(output_files)} files")
        except Exception as e:
            print(f"Error processing segmentation type {seg_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing with next segmentation type...")
    
    print(f"\nGIF visualization complete. Results saved in {master_output_dir}")
    return master_output_dir

if __name__ == "__main__":
    # Run CAM visualization for segmentation types 10, 11, 12, and 13
    output_dir = run_cam_visualization(seg_types=[10, 11, 12, 13], n_samples=4)
    print(f"All GIF visualizations complete! Check the results at: {output_dir}") 