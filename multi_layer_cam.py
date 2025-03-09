"""
Multi-Layer Attention Visualization with Consistent Grayscale Values
Supports multiple samples from different bounding boxes for cross-sample comparison
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from synapse import config, SynapseDataset, Synapse3DProcessor
from inference import VGG3D, load_and_prepare_data
from synapse.data.dataloader import normalize_cube_globally

def normalize_globally(array):
    """Apply global min-max normalization to ensure consistent grayscale values"""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val > min_val:
        normalized = (array - min_val) / (max_val - min_val)
    else:
        normalized = array.copy()
    return normalized

# Define SimpleGradCAM class for attention visualization outside main for importing
class SimpleGradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.model.eval()
        
        # Get the target layer
        parts = layer_name.split('.')
        target = self.model
        for part in parts:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)
                
        self.target_layer = target
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(
            lambda module, input, output: setattr(self, 'activations', output.clone())
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: setattr(self, 'gradients', grad_out[0].clone())
        )
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    def generate_cam(self, input_tensor):
        # Create a copy of the input tensor to avoid in-place issues
        input_copy = input_tensor.clone()
        
        # Run model through features only
        with torch.set_grad_enabled(True):
            feature_output = self.model.features(input_copy)
            
            # Create a simple classifier proxy
            pooled = F.adaptive_avg_pool3d(feature_output, (1, 1, 1))
            flattened = pooled.view(1, -1)
            
            # Dummy weights for binary classification
            fc_weight = torch.ones(2, flattened.shape[1], device=input_copy.device) * 0.01
            fc_weight[1] = 0.02  # Class 1 slightly different
            
            # Forward pass
            output = F.linear(flattened, fc_weight)
            
            # Get class 1 gradient
            self.model.zero_grad()
            
            one_hot = torch.zeros_like(output)
            one_hot[0, 1] = 1  # Target class 1
            
            output.backward(gradient=one_hot, retain_graph=True)
        
        # Make sure we have valid activations and gradients
        if self.activations is None or self.gradients is None:
            raise ValueError("No activations or gradients captured. Check hook setup.")
            
        # Get feature maps and gradients with explicit cloning
        feature_maps = self.activations.clone().detach()
        gradients = self.gradients.clone().detach()
        
        print(f"Feature maps shape: {feature_maps.shape}")
        print(f"Gradients shape: {gradients.shape}")
        
        # Global average pool gradients - avoid any potential in-place issues
        weights = gradients.mean(dim=(0, 2, 3, 4)).clone()
        
        # Create CAM by weighted sum of feature maps
        cam = torch.zeros(feature_maps.shape[2:], device=feature_maps.device)
        for i, w in enumerate(weights):
            # Use fresh clones at each step to avoid any gradient issues
            cam = cam + (w * feature_maps[0, i].clone())
            
        # ReLU and normalize - explicitly avoid in-place operations
        cam = F.relu(cam)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam

def process_single_sample(model, dataset, sample_idx, device, layers, layer_names):
    """Process a single sample and generate attention maps for all specified layers"""
    print(f"\nProcessing sample {sample_idx}...")
    
    # Get the sample
    pixel_values, syn_info, bbox_name = dataset[sample_idx]
    print(f"Sample shape: {pixel_values.shape}, BBox: {bbox_name}")
    
    # Check if this is actually a different bounding box
    bbox_info = f"Sample {sample_idx}: {syn_info.get('Var1', 'Unknown')} (BBox: {bbox_name})"
    
    # Convert to batch format and move to device
    input_tensor = pixel_values.float().unsqueeze(0).to(device)
    
    # Double-check dimensions - the model expects [B, C, D, H, W]
    if input_tensor.shape[1] != 1:  # If channels dimension is not 1
        # Probably in format [B, D, C, H, W], need to permute
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Get original image for display and apply global normalization
    original_img = pixel_values.cpu().numpy()
    img_depth = original_img.shape[0]
    
    # Apply global normalization to original image to ensure consistent grayscale values
    orig_min = np.min(original_img)
    orig_max = np.max(original_img)
    if orig_max > orig_min:
        original_img = (original_img - orig_min) / (orig_max - orig_min)
    print(f"Applied global normalization to original image: min={orig_min:.4f}, max={orig_max:.4f}")
    
    # Store results for each layer
    layer_results = {}
    
    # Process each layer
    for layer_name in layers:
        print(f"\nProcessing layer: {layer_name}")
        
        # Initialize Grad-CAM
        grad_cam = SimpleGradCAM(model, layer_name)
        
        # Generate CAM
        cam = grad_cam.generate_cam(input_tensor)
        print(f"Generated CAM with shape: {cam.shape}")
        
        # Remove hooks to prevent interference with next layer
        grad_cam.remove_hooks()
        
        # Resize CAM to match original depth if needed
        if cam.shape[0] != img_depth:
            print(f"Resizing CAM from {cam.shape} to match original depth {img_depth}")
            cam_resized = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(img_depth, cam.shape[1], cam.shape[2]),
                mode='trilinear',
                align_corners=False
            )
            cam = cam_resized.squeeze(0).squeeze(0)
        
        # Ensure consistency across slices after resizing
        min_val = cam.min()
        max_val = cam.max()
        if max_val > min_val:
            cam = (cam - min_val) / (max_val - min_val)
        
        # Store the results for this layer
        layer_results[layer_name] = cam.cpu().numpy()
    
    return {
        'original_img': original_img,
        'layer_results': layer_results,
        'bbox_name': bbox_name,
        'syn_info': syn_info,
        'bbox_info': bbox_info
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Layer Attention Visualization for Random Samples')
    parser.add_argument('--n_samples', type=int, default=3,
                      help='Number of random samples to process (default: 3)')
    parser.add_argument('--output_dir', type=str, default='results/random_samples_cam',
                      help='Directory to save output visualizations')
    parser.add_argument('--n_slices', type=int, default=4,
                      help='Number of slices to visualize per sample')
    parser.add_argument('--specific_indices', type=int, nargs='+', default=None,
                      help='Process specific sample indices instead of random ones')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing model...")
    model = VGG3D()
    model.eval()
    
    # Load data
    print("Loading data...")
    vol_data_dict, syn_df = load_and_prepare_data(config)
    processor = Synapse3DProcessor(size=config.size)
    
    # Set normalization settings - explicitly enforce volume-wide normalization
    if hasattr(processor, 'normalize_volume'):
        processor.normalize_volume = True
        print("Set processor.normalize_volume = True")
    
    # Create dataset with explicit volume-wide normalization
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        normalize_across_volume=True  # Ensure volume-wide normalization
    )
    print("Created dataset with normalize_across_volume=True")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Further ensure that the dataloader uses volume-wide normalization
    data_loader = dataset.data_loader
    if data_loader is None:
        from synapse import SynapseDataLoader
        data_loader = SynapseDataLoader("", "", "")
        dataset.data_loader = data_loader
        print("Created new dataloader for the dataset")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define layers to analyze (early, middle, late)
    layers = ["features.3", "features.6", "features.9", "features.20", "features.27"]
    layer_names = ["Early (Layer 3)", "Early-Mid (Layer 6)", "Mid-Low (Layer 9)", "Mid-High (Layer 20)", "Late (Layer 27)"]
    layer_name_map = dict(zip(layers, layer_names))
    
    # Determine which samples to process
    if args.specific_indices is not None:
        sample_indices = args.specific_indices
        print(f"Using specified indices: {sample_indices}")
    else:
        # Generate random sample indices
        sample_indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
        print(f"Generated random sample indices: {sample_indices}")
    
    # Process each sample
    for sample_idx in sample_indices:
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} exceeds dataset size ({len(dataset)}). Skipping.")
            continue
            
        # Process this sample
        result = process_single_sample(model, dataset, sample_idx, device, layers, layer_names)
        
        # Create visualization for this sample
        bbox_name = result['bbox_name']
        synapse_id = result['syn_info'].get('Var1', f'Sample_{sample_idx}')
        
        # Choose a subset of slices to display
        n_slices = min(args.n_slices, result['original_img'].shape[0])
        slice_indices = np.linspace(0, result['original_img'].shape[0] - 1, n_slices, dtype=int)
        
        # Create figure with one row per slice
        n_cols = len(layers) + 1  # Original + each layer
        
        fig, axes = plt.subplots(n_slices, n_cols, figsize=(n_cols * 4, n_slices * 3))
        
        # If we only have one row, wrap in a 2D array for consistent indexing
        if n_slices == 1:
            axes = np.array([axes])
        
        # Display slices
        for slice_idx, slice_num in enumerate(slice_indices):
            # Display original slice in first column
            axes[slice_idx, 0].set_title(f"Original (Slice {slice_num})", fontsize=10)
            axes[slice_idx, 0].imshow(result['original_img'][slice_num, 0], cmap='gray', vmin=0, vmax=1)
            axes[slice_idx, 0].axis('off')
            
            # Display attention maps for each layer
            for col, layer in enumerate(layers, start=1):
                cam = result['layer_results'][layer]
                
                # Get layer name for title (only first row)
                if slice_idx == 0:
                    axes[slice_idx, col].set_title(layer_name_map[layer], fontsize=10)
                
                # Get CAM for this slice
                cam_slice = cam[slice_num]
                
                # Upscale if needed to match original image dimensions
                if cam_slice.shape[0] != result['original_img'].shape[2] or cam_slice.shape[1] != result['original_img'].shape[3]:
                    from scipy.ndimage import zoom
                    zoom_factors = (result['original_img'].shape[2] / cam_slice.shape[0], 
                                   result['original_img'].shape[3] / cam_slice.shape[1])
                    cam_slice = zoom(cam_slice, zoom_factors, order=1)
                
                # Create overlay with consistent normalization
                ax = axes[slice_idx, col]
                # Force vmin=0, vmax=1 for consistent display
                ax.imshow(result['original_img'][slice_num, 0], cmap='gray', vmin=0, vmax=1)
                # For overlay, also ensure consistent color range
                im = ax.imshow(cam_slice, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                ax.axis('off')
                
                # Add colorbar to last column of last row
                if col == len(layers) and slice_idx == n_slices - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
        
        # Add title with sample information
        fig.suptitle(f"Multi-layer Attention Analysis - Sample: {synapse_id} (BBox: {bbox_name})", fontsize=16)
        
        # Save figure for this sample
        output_file = os.path.join(args.output_dir, f"attention_sample_{sample_idx}_bbox_{bbox_name}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention map for sample {sample_idx} saved to {output_file}")

if __name__ == "__main__":
    main() 