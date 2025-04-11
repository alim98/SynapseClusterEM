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
from synapse import config
from newdl.dataset3 import SynapseDataset
from newdl.dataloader3 import Synapse3DProcessor, SynapseDataLoader
from inference import VGG3D, load_and_prepare_data

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
    def __init__(self, model, layer_identifier):
        """
        Initialize the SimpleGradCAM object
        
        Args:
            model: The PyTorch model
            layer_identifier: Either an integer layer index or a string layer name
        """
        self.model = model
        self.model.eval()
        
        # Get the target layer
        if isinstance(layer_identifier, int):
            # If it's an integer, assume it's a direct index
            target = self.model
            if hasattr(target, 'features') and layer_identifier < len(target.features):
                target = target.features[layer_identifier]
            else:
                raise ValueError(f"Layer index {layer_identifier} out of range")
        else:
            # If it's a string, parse it as a path
            parts = layer_identifier.split('.')
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
    
    try:
        # Get the sample using dataset's __getitem__ which properly applies segmentation and alpha
        # This ensures we get the exact same processed image as defined in the config
        sample_data = dataset[sample_idx]
        
        # Check if sample was discarded (returned as None)
        if sample_data is None:
            print(f"Sample {sample_idx} was discarded during dataset processing. Skipping.")
            return None
            
        pixel_values, syn_info, bbox_name = sample_data
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
        
        # Store the original tensor for visualization - this will include any masking from alpha
        # and segmentation settings that were applied when the dataset created this sample
        original_img = input_tensor.clone().cpu().numpy()
        img_depth = original_img.shape[2]  # [B, C, D, H, W]
        
        print(f"Using dataset-processed image with alpha={config.alpha} and segmentation_type={config.segmentation_type}")
        
        # Store results for each layer
        layer_results = {}
        
        # Process each layer
        for i, layer_idx in enumerate(layers):
            # Get layer name if provided
            layer_name = layer_names[i] if i < len(layer_names) else f"layer{layer_idx}"
            print(f"\nProcessing layer: {layer_idx}")
            
            # Initialize Grad-CAM
            grad_cam = SimpleGradCAM(model, layer_idx)
            
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
    except Exception as e:
        # Print the error and allow the caller to handle it
        print(f"Error processing sample {sample_idx}: {str(e)}")
        raise

def visualize_samples_attention(model, dataset, sample_indices, output_dir, layers, layer_names, n_slices=4):
    """
    Visualizes attention maps for a set of samples as animated GIFs.
    
    Args:
        model: The PyTorch model
        dataset: The dataset containing the samples
        sample_indices: List of sample indices to visualize
        output_dir: Directory to save visualizations
        layers: List of layer indices to visualize
        layer_names: List of layer names corresponding to the indices
        n_slices: Number of slices to visualize per sample (used for static images only)
        
    Returns:
        List of paths to saved visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Attempt to import imageio for GIF creation
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not found. Installing imageio via pip...")
        import subprocess
        subprocess.check_call(["pip", "install", "imageio", "pillow"])
        import imageio
    
    output_files = []
    
    # Create a mapping from layer numbers to readable names
    layer_name_map = {}
    for i, layer in enumerate(layers):
        layer_name = layer_names[i] if i < len(layer_names) else f"Layer {layer}"
        layer_name_map[layer] = layer_name
    
    # Process each sample
    for sample_idx in sample_indices:
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} exceeds dataset size ({len(dataset)}). Skipping.")
            continue
        
        try:
            # Process this sample
            result = process_single_sample(model, dataset, sample_idx, device, layers, layer_names)
            
            # Skip if sample was discarded
            if result is None:
                print(f"Sample {sample_idx} was discarded. Skipping visualization.")
                continue
            
            # Create visualization for this sample
            bbox_name = result['bbox_name']
            synapse_id = result['syn_info'].get('Var1', f'Sample_{sample_idx}')
            
            # Determine the number of slices to include in the GIF
            # Use all slices from the volume for the GIF
            total_slices = result['original_img'].shape[2]
            step = max(1, total_slices // 40)  # Limit to about 40 frames for smooth animation
            slice_indices = range(0, total_slices, step)
            
            # Create a separate GIF for each layer
            for layer_idx, layer_name in enumerate(result['layer_results'].keys()):
                # Get display name for this layer
                display_name = layer_name_map.get(layer_name, layer_name)
                
                # Get the CAM data for this layer
                cam = result['layer_results'][layer_name]
                
                # Create a temporary directory for storing frames
                import tempfile
                temp_dir = tempfile.mkdtemp()
                
                # Create frames for the GIF
                frames = []
                
                # For each slice
                for frame_idx, slice_num in enumerate(slice_indices):
                    # Create a new figure for this frame
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Get CAM for this slice
                    cam_slice = cam[slice_num]
                    
                    # Upscale if needed to match original image dimensions
                    if cam_slice.shape[0] != result['original_img'].shape[3] or cam_slice.shape[1] != result['original_img'].shape[4]:
                        from scipy.ndimage import zoom
                        zoom_factors = (result['original_img'].shape[3] / cam_slice.shape[0], 
                                      result['original_img'].shape[4] / cam_slice.shape[1])
                        cam_slice = zoom(cam_slice, zoom_factors, order=1)
                    
                    # Display the original image
                    ax.imshow(result['original_img'][0, 0, slice_num, :, :], cmap='gray', vmin=0, vmax=1)
                    
                    # Overlay the attention map
                    im = ax.imshow(cam_slice, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                    ax.axis('off')
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    
                    # Add slice information
                    plt.title(f"Sample {sample_idx}: {display_name} - Slice {slice_num}/{total_slices-1}", fontsize=12)
                    
                    # Save this frame
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:03d}.png")
                    plt.tight_layout()
                    plt.savefig(frame_path, dpi=100)
                    plt.close()
                    
                    # Add to frames list
                    frames.append(frame_path)
                
                # Create GIF
                alpha_value = config.alpha if hasattr(config, 'alpha') else 'unknown'
                seg_type = config.segmentation_type if hasattr(config, 'segmentation_type') else 'unknown'
                gif_file = os.path.join(output_dir, f"attention_sample_{sample_idx}_bbox_{bbox_name}_{display_name}_alpha{alpha_value}_seg{seg_type}.gif")
                
                # Read all frames and create GIF
                images = [imageio.imread(frame) for frame in frames]
                
                # Save with optimized settings - lower duration for smoother animation
                imageio.mimsave(gif_file, images, duration=0.15, loop=0)  # loop=0 means infinite loop
                
                print(f"Attention GIF for sample {sample_idx}, layer {display_name} saved to {gif_file}")
                output_files.append(gif_file)
                
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return output_files

def visualize_cluster_attention(model, dataset, clusters_samples, output_dir, layers, layer_names, n_slices=3):
    """
    Visualizes attention maps for samples grouped by clusters.
    
    Args:
        model: The PyTorch model
        dataset: The dataset containing the samples
        clusters_samples: Dictionary mapping cluster IDs to lists of sample indices
        output_dir: Directory to save visualizations
        layers: List of layer indices to visualize
        layer_names: List of layer names corresponding to the indices
        n_slices: Number of slices to visualize per sample
        
    Returns:
        Dictionary mapping cluster IDs to lists of visualization paths
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    for cluster_id, samples in clusters_samples.items():
        print(f"Processing cluster {cluster_id}...")
        # Create a subdirectory for this cluster
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Extract sample indices from DataFrame rows
        sample_indices = []
        for sample in samples:
            try:
                if hasattr(sample, 'name'):  # If it's a Series
                    sample_indices.append(sample.name)
                elif hasattr(sample, 'iloc'):  # If it's a DataFrame row
                    sample_indices.append(sample.index[0])
                else:  # Assume it's already an index
                    sample_indices.append(sample)
            except Exception as e:
                print(f"Error extracting index from sample: {e}")
                continue
        
        # Visualize samples for this cluster
        output_files = visualize_samples_attention(
            model, dataset, sample_indices, cluster_dir, 
            layers, layer_names, n_slices
        )
        
        results[cluster_id] = output_files
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Layer Attention Visualization as Animated GIFs')
    parser.add_argument('--n_samples', type=int, default=10,
                      help='Number of random samples to process (default: 3)')
    parser.add_argument('--output_dir', type=str, default='results/random_samples_cam',
                      help='Directory to save output visualizations')
    parser.add_argument('--n_slices', type=int, default=4,
                      help='Number of slices to visualize per sample (for static images - ignored for GIFs)')
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
   
    # Create dataset with explicit volume-wide normalization
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        normalize_across_volume=True  # Ensure volume-wide normalization
    )
    print(f"Created dataset with normalize_across_volume=True, alpha={config.alpha}, segmentation_type={config.segmentation_type}")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Further ensure that the dataloader uses volume-wide normalization
    data_loader = dataset.data_loader
    if data_loader is None:
        from newdl.dataloader3 import SynapseDataLoader
        data_loader = SynapseDataLoader("", "", "")
        dataset.data_loader = data_loader
        print("Created new dataloader for the dataset")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define layers to analyze (early, middle, late)
    layers = ["features.20"]
    layer_names = ["Mid-High (Layer 20)"]
    
    # Determine which samples to process
    if args.specific_indices is not None:
        sample_indices = args.specific_indices
        print(f"Using specified indices: {sample_indices}")
    else:
        # Generate random sample indices
        sample_indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
        print(f"Generated random sample indices: {sample_indices}")
    
    # Use the visualize_samples_attention function for consistent visualization
    output_files = visualize_samples_attention(
        model=model,
        dataset=dataset,
        sample_indices=sample_indices,
        output_dir=args.output_dir,
        layers=layers,
        layer_names=layer_names,
        n_slices=args.n_slices
    )
    
    print(f"Processing complete. {len(output_files)} animated GIFs generated.")
    
    return output_files

if __name__ == "__main__":
    main() 