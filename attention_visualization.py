#!/usr/bin/env python
"""
Model Attention Visualization for Synapse Analysis

This script visualizes where the VGG3D model attends in input synapse images
using Gradient-weighted Class Activation Mapping (Grad-CAM).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom
from pathlib import Path
import argparse
import pandas as pd
import traceback

from synapse import (
    SynapseDataLoader, 
    Synapse3DProcessor, 
    SynapseDataset, 
    config
)
from synapse.models.vgg3d import Vgg3D, load_model_from_checkpoint
from inference import VGG3D, load_and_prepare_data

class GradCAM3D:
    """
    Grad-CAM implementation for 3D CNN models.
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        # Extract the target layer from model using the provided layer name
        layers = target_layer_name.split('.')
        target_module = self.model
        for layer in layers:
            if layer.isdigit():
                target_module = target_module[int(layer)]
            else:
                target_module = getattr(target_module, layer)
        
        self.target_layer = target_module
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.register_hooks()
        
    def register_hooks(self):
        """
        Register forward and backward hooks on the target layer.
        """
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def _forward_features(self, x):
        """Run the model features only up to the classifier"""
        # Get all features layers
        features = self.model.features
        # Manually run through the layers
        for i, layer in enumerate(features):
            x = layer(x)
            # If we've reached a point beyond our target layer, we can stop
            if isinstance(self.target_layer, nn.Conv3d) and isinstance(layer, nn.Conv3d):
                if layer == self.target_layer:
                    break
        return x
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate class activation map for the target class.
        
        Args:
            input_tensor: Input tensor to the model
            target_class: Target class for visualization (None = model prediction)
            
        Returns:
            Normalized class activation map
        """
        print(f"Original input dimensions: {input_tensor.shape}")
        
        # Forward pass through features only (no classifier)
        features_output = self.model.features(input_tensor)
        
        # Create a dummy classification output for backprop
        # We'll use a global average pool + linear layer
        batch_size = input_tensor.size(0)
        pooled = F.adaptive_avg_pool3d(features_output, (1, 1, 1))
        flattened = pooled.view(batch_size, -1)
        
        # Create a simple linear layer for classification
        num_channels = features_output.size(1)
        dummy_weight = torch.ones(2, num_channels, device=input_tensor.device) * 0.01
        dummy_weight[1] = 0.02  # Make class 1 slightly different
        
        # Generate dummy logits
        logits = F.linear(flattened, dummy_weight)
        pred_class = 1  # Always use class 1 for visualization
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0, pred_class] = 1
        
        # Backward pass
        logits.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations.detach()
        gradients = self.gradients.detach()
        
        print(f"Activation shape after {self.target_layer}: {activations.shape}")
        print(f"Gradients shape: {gradients.shape}")
        
        # Weight the channels by corresponding gradients
        # Take the mean of the gradients across batch dimension (0) and spatial dimensions (2,3,4)
        # For a 5D tensor of shape [N, C, D, H, W], we want the mean of all dims except channels (1)
        weights = torch.mean(gradients, dim=(0, 2, 3, 4))  # This gives a 1D tensor of size [C]
        
        # Reshape weights for broadcasting: [C] -> [C, 1, 1, 1]
        weights = weights.reshape(-1, 1, 1, 1)  # Channel weights: [C, 1, 1, 1]
        
        print(f"Channel weights shape: {weights.shape}")
        
        # Calculate weighted activations - preserving 3D structure
        # Multiply each channel by its weight and sum over channels
        # This gives us a 3D map of shape [D, H, W]
        cam = torch.zeros(activations.shape[2:], device=activations.device)
        for i, w in enumerate(weights.flatten()):
            cam += w * activations[0, i]
        
        print(f"CAM shape after weighted sum: {cam.shape}")
        
        # Apply ReLU to focus on positive attributions
        cam = F.relu(cam)
        
        # Normalize CAM
        if cam.max() > 0:
            cam = cam / cam.max()
        
        print(f"Final CAM shape: {cam.shape}")
        
        return cam, pred_class

def visualize_attention(model, sample_idx, output_dir, n_slices=5, target_class=None, target_layer_name='features.3'):
    """
    Visualize model attention using Grad-CAM.
    
    Args:
        model: The model to visualize attention for
        sample_idx: Index of the sample to visualize
        output_dir: Directory to save attention maps
        n_slices: Number of slices to visualize
        target_class: Target class for visualization (None = model prediction)
        target_layer_name: Name of the layer to use for Grad-CAM
    
    Returns:
        output_file: Path to the saved attention map
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Visualizing attention for sample index {sample_idx}")
        
        # Get data sample
        sample = get_sample(sample_idx)
        
        # Prepare input tensor
        input_tensor = prepare_input(sample['image'])
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Create a simplified GradCAM implementation based on our debug_cam.py findings
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
                self.target_layer.register_forward_hook(
                    lambda module, input, output: setattr(self, 'activations', output)
                )
                self.target_layer.register_full_backward_hook(
                    lambda module, grad_in, grad_out: setattr(self, 'gradients', grad_out[0])
                )
                
            def generate_cam(self, input_tensor):
                # Run model through features only
                feature_output = self.model.features(input_tensor)
                
                # Create a simple classifier proxy (GAP + FC) to generate gradients
                pooled = F.adaptive_avg_pool3d(feature_output, (1, 1, 1))
                flattened = pooled.view(1, -1)
                
                # Dummy weights for binary classification
                fc_weight = torch.ones(2, flattened.shape[1], device=input_tensor.device) * 0.01
                fc_weight[1] = 0.02  # Class 1 slightly different
                
                # Forward pass through proxy classifier
                output = F.linear(flattened, fc_weight)
                
                # Get class 1 gradient
                self.model.zero_grad()
                
                one_hot = torch.zeros_like(output)
                one_hot[0, 1] = 1  # Target class 1
                
                output.backward(gradient=one_hot, retain_graph=True)
                
                # Get feature maps and gradients
                feature_maps = self.activations.detach()
                gradients = self.gradients.detach()
                
                print(f"Feature maps shape: {feature_maps.shape}")
                print(f"Gradients shape: {gradients.shape}")
                
                # Global average pool gradients across batch and spatial dimensions
                weights = gradients.mean(dim=(0, 2, 3, 4))
                
                # Create CAM by weighted sum of feature maps
                cam = torch.zeros(feature_maps.shape[2:], device=feature_maps.device)
                for i, w in enumerate(weights):
                    cam += w * feature_maps[0, i]
                    
                # ReLU and normalize
                cam = F.relu(cam)
                if cam.max() > 0:
                    cam = cam / cam.max()
                    
                return cam
        
        # Initialize Grad-CAM
        grad_cam = SimpleGradCAM(model, target_layer_name)
        
        # Generate class activation map
        cam = grad_cam.generate_cam(input_tensor)
        print(f"Generated CAM with shape: {cam.shape}")
        
        # Get original image dimensions
        if isinstance(sample['image'], torch.Tensor):
            img_depth = sample['image'].shape[0]
        else:  # Numpy array
            img_depth = sample['image'].shape[0]
        
        # If CAM has fewer depth slices than the original image, resize it
        if cam.shape[0] != img_depth:
            print(f"Resizing CAM from {cam.shape} to match original depth {img_depth}")
            # Create a new tensor for the resized CAM
            cam_resized = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1, 1, D, H, W]
                size=(img_depth, cam.shape[1], cam.shape[2]),  # Target size
                mode='trilinear',
                align_corners=False
            )
            cam = cam_resized.squeeze(0).squeeze(0)  # Remove batch and channel dims
        
        print(f"Final CAM shape after resizing: {cam.shape}")
        
        # Create a multi-row figure with original, heatmap and overlay
        n_cols = 3  # Original, Heatmap, Overlay
        if n_slices > img_depth:
            n_slices = img_depth
            
        # Select slices to visualize
        slice_indices = np.linspace(0, img_depth - 1, n_slices, dtype=int)
        
        # Create figure
        fig, axes = plt.subplots(n_slices, n_cols, figsize=(n_cols * 4, n_slices * 4))
        
        # Get original image
        original_img = sample['image']
        if isinstance(original_img, torch.Tensor):
            original_img = original_img.squeeze(1).cpu().numpy()  # Remove channel dimension
        else:
            original_img = original_img.squeeze(1)  # Remove channel dimension if it exists
            
        # Get metadata
        synapse_id = sample['syn_info'].get('Var1', 'Unknown')
        bbox_name = sample['bbox_name']
            
        # Plot title
        fig.suptitle(f"Sample: {synapse_id} (BBox: {bbox_name})", fontsize=16)
        
        # Column titles
        col_titles = ['Original', 'Attention Heatmap', 'Overlay']
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=14)
        
        # Visualize each slice
        for row, slice_idx in enumerate(slice_indices):
            # Get the slice image
            img_slice = original_img[slice_idx]
            
            # Get the attention map for this slice
            cam_slice = cam[slice_idx].cpu().numpy()
            
            # Plot original image
            axes[row, 0].imshow(img_slice, cmap='gray')
            axes[row, 0].set_title(f"Original (Slice {slice_idx})")
            axes[row, 0].axis('off')
            
            # Plot attention heatmap
            im = axes[row, 1].imshow(cam_slice, cmap='jet')
            # Add a colorbar to the heatmap
            divider = make_axes_locatable(axes[row, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            axes[row, 1].set_title("Attention Heatmap")
            axes[row, 1].axis('off')
            
            # Plot overlay
            axes[row, 2].imshow(img_slice, cmap='gray')
            axes[row, 2].imshow(cam_slice, cmap='jet', alpha=0.5)
            axes[row, 2].set_title("Overlay")
            axes[row, 2].axis('off')
        
        # Save figure
        output_file = os.path.join(output_dir, f"attention_sample_{sample_idx}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention map saved as {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"Error visualizing attention: {e}")
        traceback.print_exc()
        return None

def process_multiple_samples(model, dataset, sample_indices, output_dir, n_slices=3, target_class=None, target_layer='features.3'):
    """
    Process multiple samples and visualize attention maps for each.
    
    Args:
        model: The model to visualize attention for
        dataset: The dataset containing the samples
        sample_indices: List of sample indices to process
        output_dir: Directory to save attention maps
        n_slices: Number of slices to visualize
        target_class: Target class for visualization (None = model prediction)
        target_layer: Name of the layer to use for Grad-CAM
    
    Returns:
        List of output files
    """
    print(f"Processing {len(sample_indices)} samples")
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    for idx in sample_indices:
        try:
            output_file = visualize_attention(model, idx, output_dir, n_slices, target_class, target_layer)
            output_files.append(output_file)
            print(f"Processed sample {idx}, saved to {output_file}")
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
    
    return output_files

def identify_top_attended_regions(model, dataset, n_samples=10, output_dir="results/top_attention", n_top_regions=5, target_layer='features.3'):
    """
    Identify and visualize top attended regions across multiple samples.
    
    Args:
        model: The model to visualize attention for
        dataset: The dataset containing the samples
        n_samples: Number of samples to process
        output_dir: Directory to save visualizations
        n_top_regions: Number of top regions to identify
        target_layer: Name of the layer to use for Grad-CAM
    
    Returns:
        Path to the saved visualization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize GradCAM
    grad_cam = GradCAM3D(model, target_layer)
    
    # Initialize list to store attention statistics
    attention_stats = []
    
    # Process each sample
    sample_indices = list(range(min(n_samples, len(dataset))))
    for idx in sample_indices:
        try:
            # Get sample
            pixel_values, syn_info, bbox_name = dataset[idx]
            
            # Prepare input tensor
            if isinstance(pixel_values, torch.Tensor):
                input_tensor = pixel_values.float().unsqueeze(0).to(device)
            else:
                input_tensor = torch.from_numpy(pixel_values).float().unsqueeze(0).to(device)
            
            # Generate CAM
            cam, predicted_class = grad_cam.generate_cam(input_tensor)
            
            # Calculate attention statistics
            mean_attn = cam.mean().item()
            max_attn = cam.max().item()
            
            # Get the coordinates of the maximum attention
            max_coords = np.unravel_index(cam.argmax(), cam.shape)
            
            # Store statistics
            attention_stats.append({
                'sample_idx': idx,
                'bbox_name': bbox_name,
                'syn_info': syn_info,
                'mean_attn': mean_attn,
                'max_attn': max_attn,
                'max_coords': max_coords,
                'cam': cam
            })
            
            print(f"Processed sample {idx}: mean={mean_attn:.4f}, max={max_attn:.4f}, max_coords={max_coords}")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
    
    # Sort by maximum attention
    attention_stats.sort(key=lambda x: x['max_attn'], reverse=True)
    
    # Get top regions
    top_regions = attention_stats[:n_top_regions]
    
    # Visualize top regions
    fig, axes = plt.subplots(n_top_regions, 3, figsize=(15, 5*n_top_regions))
    
    if n_top_regions == 1:
        axes = axes.reshape(1, -1)
    
    for i, region in enumerate(top_regions):
        idx = region['sample_idx']
        cam = region['cam']
        
        # Get sample data
        pixel_values, _, _ = dataset[idx]
        
        # Get the center slice around the max attention point
        if isinstance(pixel_values, torch.Tensor):
            depth = pixel_values.shape[0]
        else:
            depth = pixel_values.shape[0]
        
        # Get indices of the maximum attention
        d_idx = region['max_coords'][0]
        
        # Ensure index is within bounds
        d_idx = max(0, min(d_idx, depth-1))
        
        # Get the slice
        if isinstance(pixel_values, torch.Tensor):
            orig_slice = pixel_values[d_idx].squeeze().cpu().numpy()
        else:
            orig_slice = pixel_values[d_idx].squeeze()
        
        # Get the CAM slice
        cam_slice = cam[d_idx]
        
        # Plot original
        axes[i, 0].imshow(orig_slice, cmap='gray')
        axes[i, 0].set_title(f"Sample {idx} (Slice {d_idx})")
        axes[i, 0].axis('off')
        
        # Plot CAM
        im = axes[i, 1].imshow(cam_slice, cmap='jet')
        axes[i, 1].set_title("Attention Heatmap")
        axes[i, 1].axis('off')
        
        # Add colorbar
        divider = make_axes_locatable(axes[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Plot overlay
        axes[i, 2].imshow(orig_slice, cmap='gray')
        axes[i, 2].imshow(cam_slice, cmap='jet', alpha=0.5)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
    
    # Save figure
    output_file = os.path.join(output_dir, "top_attended_regions.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return top_regions

def inspect_model_architecture(model):
    """
    Inspect the model architecture to understand each layer's shape transformation.
    
    Args:
        model: The PyTorch model to inspect
    """
    print("\nInspecting Model Architecture:")
    print("=" * 50)
    
    # Create a dummy input to trace through the model
    dummy_input = torch.randn(1, 16, 1, 80, 80)  # [batch, depth, channels, height, width]
    # Convert to the format expected by the model
    dummy_input = dummy_input.permute(0, 2, 1, 3, 4)  # [batch, channels, depth, height, width]
    
    # Print input shape
    print(f"Input shape: {dummy_input.shape}")
    
    # Inspect feature layers
    if hasattr(model, 'features') and isinstance(model.features, torch.nn.Sequential):
        with torch.no_grad():
            x = dummy_input
            print("\nFeature Layers:")
            for i, layer in enumerate(model.features):
                x = layer(x)
                print(f"  Layer {i} ({type(layer).__name__}): {x.shape}")
                
                # After MaxPool3D, check if we've lost the depth dimension
                if isinstance(layer, torch.nn.MaxPool3d) and x.shape[2] == 1:
                    print(f"  --> Depth dimension collapsed to 1 at layer {i}")
    
    # Inspect classifier layers
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        with torch.no_grad():
            # Flatten the output from features
            x = model.features(dummy_input)
            x = x.view(x.size(0), -1)
            print(f"\nInput to classifier: {x.shape}")
            
            print("\nClassifier Layers:")
            for i, layer in enumerate(model.classifier):
                x = layer(x)
                print(f"  Layer {i} ({type(layer).__name__}): {x.shape}")
    
    print("=" * 50)

def get_sample(sample_idx):
    """
    Get a sample from the dataset at the specified index.
    
    Args:
        sample_idx: Index of the sample to get
        
    Returns:
        Dictionary containing the sample data
    """
    # Create dataset if not already created
    print("Creating dataset")
    
    # Load data using the same import as in the main file
    vol_data_dict, syn_df = load_and_prepare_data(config)
    processor = Synapse3DProcessor(size=config.size)
    
    # Create dataset
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get the sample
    print(f"Getting sample {sample_idx} from dataset")
    pixel_values, syn_info, bbox_name = dataset[sample_idx]
    print(f"Sample shape: {pixel_values.shape}, bbox: {bbox_name}")
    
    # Return as a dictionary
    return {
        'image': pixel_values,
        'syn_info': syn_info,
        'bbox_name': bbox_name
    }

def prepare_input(pixel_values):
    """
    Prepare the input tensor for the model.
    
    Args:
        pixel_values: The pixel values to prepare
        
    Returns:
        Prepared input tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensor and move to device
    print("Converting to tensor")
    if isinstance(pixel_values, torch.Tensor):
        input_tensor = pixel_values.float().unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(pixel_values).float().unsqueeze(0).to(device)
    
    print(f"Input tensor shape before permute: {input_tensor.shape}")
    
    # Permute dimensions from [batch, depth, channels, height, width] to [batch, channels, depth, height, width]
    # This is what the model expects for Conv3D layers
    input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
    
    print(f"Input tensor shape after permute: {input_tensor.shape}")
    
    return input_tensor

def main():
    try:
        parser = argparse.ArgumentParser(description='Visualize model attention using Grad-CAM')
        parser.add_argument('--seg_type', type=int, default=config.segmentation_type, 
                        choices=range(0, 11), help='Type of segmentation overlay (0-10)')
        parser.add_argument('--alpha', type=float, default=config.alpha,
                        help='Alpha value for overlaying segmentation')
        parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample to visualize')
        parser.add_argument('--output_dir', type=str, default='results/attention_maps',
                        help='Directory to save attention maps')
        parser.add_argument('--n_slices', type=int, default=5,
                        help='Number of slices to visualize')
        parser.add_argument('--target_class', type=int, default=None,
                        help='Target class for visualization (None = model prediction)')
        parser.add_argument('--batch_mode', action='store_true',
                        help='Process multiple samples in batch mode')
        parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to process in batch mode')
        parser.add_argument('--find_top_regions', action='store_true',
                        help='Find and visualize top attended regions across samples')
        parser.add_argument('--n_top_regions', type=int, default=5,
                        help='Number of top regions to identify')
        parser.add_argument('--inspect_model', action='store_true',
                        help='Inspect model architecture')
        parser.add_argument('--target_layer', type=str, default='features.3',
                        help='Target layer for Grad-CAM visualization')
        parser.add_argument('--multi_layer', action='store_true',
                        help='Visualize attention at multiple network layers')
        
        args = parser.parse_args()
        
        print(f"Starting attention visualization with args: {args}")
        
        # Initialize VGG3D model
        print("Initializing VGG3D model")
        model = VGG3D()
        
        # Inspect model architecture if requested
        if args.inspect_model:
            inspect_model_architecture(model)
            return
            
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
            
        # Load and prepare data
        print("Loading and preparing data")
        
        # Process a single sample with multi-layer analysis
        if args.multi_layer:
            # Define layers to analyze (early, middle, late)
            layers = ["features.3", "features.20", "features.27"]
            layer_names = ["Early (Layer 3)", "Middle (Layer 20)", "Late (Layer 27)"]
            
            # Get data sample
            sample = get_sample(args.sample_idx)
            synapse_id = sample['syn_info'].get('Var1', 'Unknown')
            bbox_name = sample['bbox_name']
            
            # Prepare input tensor
            input_tensor = prepare_input(sample['image'])
            print(f"Input tensor shape: {input_tensor.shape}")
            
            # Get original image
            original_img = sample['image']
            if isinstance(original_img, torch.Tensor):
                original_img = original_img.squeeze(1).cpu().numpy()
            else:
                original_img = original_img.squeeze(1)
            
            # Get image depth and select slices
            img_depth = original_img.shape[0]
            n_slices = min(args.n_slices, img_depth)
            slice_indices = np.linspace(0, img_depth - 1, n_slices, dtype=int)
            
            # Create figure for multi-layer comparison
            # Rows: different slices, Columns: different layers
            fig, axes = plt.subplots(n_slices, len(layers) + 1, figsize=((len(layers) + 1) * 4, n_slices * 4))
            
            # First column shows original slices
            for i, slice_idx in enumerate(slice_indices):
                # Plot original image
                axes[i, 0].imshow(original_img[slice_idx], cmap='gray')
                axes[i, 0].set_title(f"Original (Slice {slice_idx})")
                axes[i, 0].axis('off')
            
            # Create a SimpleGradCAM implementation
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
                        lambda module, input, output: setattr(self, 'activations', output)
                    )
                    self.backward_hook = self.target_layer.register_full_backward_hook(
                        lambda module, grad_in, grad_out: setattr(self, 'gradients', grad_out[0])
                    )
                    
                def remove_hooks(self):
                    self.forward_hook.remove()
                    self.backward_hook.remove()
                    
                def generate_cam(self, input_tensor):
                    # Run model through features only
                    feature_output = self.model.features(input_tensor)
                    
                    # Create a simple classifier proxy (GAP + FC) to generate gradients
                    pooled = F.adaptive_avg_pool3d(feature_output, (1, 1, 1))
                    flattened = pooled.view(1, -1)
                    
                    # Dummy weights for binary classification
                    fc_weight = torch.ones(2, flattened.shape[1], device=input_tensor.device) * 0.01
                    fc_weight[1] = 0.02  # Class 1 slightly different
                    
                    # Forward pass through proxy classifier
                    output = F.linear(flattened, fc_weight)
                    
                    # Get class 1 gradient
                    self.model.zero_grad()
                    
                    one_hot = torch.zeros_like(output)
                    one_hot[0, 1] = 1  # Target class 1
                    
                    output.backward(gradient=one_hot, retain_graph=True)
                    
                    # Get feature maps and gradients
                    feature_maps = self.activations.detach()
                    gradients = self.gradients.detach()
                    
                    print(f"Feature maps shape: {feature_maps.shape}")
                    print(f"Gradients shape: {gradients.shape}")
                    
                    # Global average pool gradients across batch and spatial dimensions
                    weights = gradients.mean(dim=(0, 2, 3, 4))
                    
                    # Create CAM by weighted sum of feature maps
                    cam = torch.zeros(feature_maps.shape[2:], device=feature_maps.device)
                    for i, w in enumerate(weights):
                        cam += w * feature_maps[0, i]
                        
                    # ReLU and normalize
                    cam = F.relu(cam)
                    if cam.max() > 0:
                        cam = cam / cam.max()
                        
                    return cam
            
            # Process each layer
            for col, (layer, layer_name) in enumerate(zip(layers, layer_names), start=1):
                print(f"\nProcessing layer: {layer}")
                
                # Initialize Grad-CAM
                grad_cam = SimpleGradCAM(model, layer)
                
                # Generate CAM
                cam = grad_cam.generate_cam(input_tensor)
                print(f"Generated CAM with shape: {cam.shape}")
                
                # Remove hooks to prevent interference with next layer
                grad_cam.remove_hooks()
                
                # If CAM has fewer depth slices than the original image, resize it
                if cam.shape[0] != img_depth:
                    print(f"Resizing CAM from {cam.shape} to match original depth {img_depth}")
                    # Create a new tensor for the resized CAM
                    cam_resized = F.interpolate(
                        cam.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1, 1, D, H, W]
                        size=(img_depth, cam.shape[1], cam.shape[2]),  # Target size
                        mode='trilinear',
                        align_corners=False
                    )
                    cam = cam_resized.squeeze(0).squeeze(0)  # Remove batch and channel dims
                
                print(f"Final CAM shape after resizing: {cam.shape}")
                
                # Display CAM for each slice
                for row, slice_idx in enumerate(slice_indices):
                    # Get CAM slice
                    cam_slice = cam[slice_idx].cpu().numpy()
                    
                    # If CAM spatial dimensions are smaller than image, upscale
                    if cam_slice.shape != original_img[slice_idx].shape:
                        cam_slice = zoom(cam_slice, 
                                         (original_img[slice_idx].shape[0] / cam_slice.shape[0], 
                                          original_img[slice_idx].shape[1] / cam_slice.shape[1]),
                                         order=1)
                    
                    # Create overlay
                    ax = axes[row, col]
                    ax.imshow(original_img[slice_idx], cmap='gray')
                    im = ax.imshow(cam_slice, cmap='jet', alpha=0.5)
                    ax.set_title(f"{layer_name}")
                    ax.axis('off')
                    
                    # Add colorbar on the rightmost column
                    if col == len(layers):
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im, cax=cax)
            
            # Add overall title
            fig.suptitle(f"Multi-layer Attention Analysis - Sample: {synapse_id} (BBox: {bbox_name})", fontsize=16)
            
            # Save figure
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, f"multi_layer_attention_sample_{args.sample_idx}.png")
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Multi-layer attention map saved as {output_file}")
            
        # Process a single sample with a specific layer
        elif not args.batch_mode and not args.find_top_regions:
            output_file = visualize_attention(
                model, 
                args.sample_idx, 
                args.output_dir, 
                args.n_slices,
                args.target_class,
                args.target_layer
            )
            print(f"Attention map saved to: {output_file}")
        
        # Process multiple samples in batch mode
        elif args.batch_mode:
            print(f"Processing {args.n_samples} samples in batch mode")
            # Create dataset
            vol_data_dict, syn_df = load_and_prepare_data(config)
            processor = Synapse3DProcessor(size=config.size)
            
            # Create dataset
            dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=syn_df,
                processor=processor,
                segmentation_type=args.seg_type,
                alpha=args.alpha
            )
            
            # Get sample indices
            sample_indices = list(range(min(args.n_samples, len(dataset))))
            
            output_files = process_multiple_samples(
                model, 
                dataset, 
                sample_indices, 
                args.output_dir, 
                args.n_slices,
                args.target_class,
                args.target_layer
            )
            print(f"Processed {len(output_files)} samples")
        
        # Find top attended regions
        elif args.find_top_regions:
            print(f"Finding top {args.n_top_regions} attended regions across {args.n_samples} samples")
            
            # Create dataset
            vol_data_dict, syn_df = load_and_prepare_data(config)
            processor = Synapse3DProcessor(size=config.size)
            
            # Create dataset
            dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=syn_df,
                processor=processor,
                segmentation_type=args.seg_type,
                alpha=args.alpha
            )
            
            top_regions = identify_top_attended_regions(
                model, 
                dataset, 
                args.n_samples, 
                args.output_dir, 
                args.n_top_regions,
                args.target_layer
            )
            print(f"Top attended regions: {top_regions}")
            
    except Exception as e:
        print(f"Error in attention visualization: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 