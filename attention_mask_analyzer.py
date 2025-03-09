"""
Attention Mask Analyzer

This module provides tools for analyzing the overlap between attention maps and masked regions in 3D
electron microscopy data.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from synapse import config, SynapseDataset, Synapse3DProcessor
from synapse.data.dataloader import SynapseDataLoader
from inference import VGG3D, load_and_prepare_data
# Import classes from multi_layer_cam.py
from multi_layer_cam import SimpleGradCAM, normalize_globally, process_single_sample


class AttentionMaskAnalyzer:
    """
    Analyzes the overlap between attention maps and masked regions in 3D electron microscopy data.
    """
    
    def __init__(self, model, dataset, segmentation_type, output_dir='results/attention_analysis'):
        """
        Initialize the analyzer.
        
        Args:
            model: The VGG3D model
            dataset: The SynapseDataset
            segmentation_type: Segmentation type to use for mask generation
            output_dir: Directory to save analysis results
        """
        self.model = model
        self.dataset = dataset
        self.segmentation_type = segmentation_type
        self.output_dir = output_dir
        self.data_loader = dataset.data_loader or SynapseDataLoader(
            raw_base_dir=config.raw_base_dir,
            seg_base_dir=config.seg_base_dir,
            add_mask_base_dir=config.add_mask_base_dir
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize layer names for analysis
        self.layers = ["features.3", "features.6", "features.9", "features.20", "features.27"]
        self.layer_names = ["Early (Layer 3)", "Early-Mid (Layer 6)", "Mid-Low (Layer 9)", 
                            "Mid-High (Layer 20)", "Late (Layer 27)"]
        self.layer_name_map = dict(zip(self.layers, self.layer_names))
        
        # For storing results
        self.results = {}
        
        # Device for running the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def _generate_attention_maps(self, input_tensor, sample_info):
        """
        Generate attention maps for all layers.
        
        Args:
            input_tensor: Input tensor for the model
            sample_info: Information about the sample
            
        Returns:
            Dictionary with attention maps for each layer
        """
        attention_maps = {}
        
        for layer_name in self.layers:
            print(f"\nProcessing layer: {layer_name}")
            
            # Initialize Grad-CAM for this layer
            grad_cam = SimpleGradCAM(self.model, layer_name)
            
            # Generate CAM
            cam = grad_cam.generate_cam(input_tensor)
            
            # Remove hooks
            grad_cam.remove_hooks()
            
            # Store the CAM
            attention_maps[layer_name] = cam.cpu().numpy()
        
        return attention_maps
    
    def _get_masks(self, raw_vol, seg_vol, add_mask_vol, sample_info, original_shape):
        """
        Extract masks based on segmentation type.
        
        Args:
            raw_vol: Raw volume data
            seg_vol: Segmentation volume data
            add_mask_vol: Additional mask volume data
            sample_info: Information about the sample
            original_shape: Original shape of the data
            
        Returns:
            Binary mask array showing the masked regions
        """
        # Extract coordinates
        central_coord = (
            sample_info.get('central_coord_1', sample_info.get('x')),
            sample_info.get('central_coord_2', sample_info.get('y')),
            sample_info.get('central_coord_3', sample_info.get('z'))
        )
        
        side1_coord = (
            sample_info.get('side_1_coord_1'),
            sample_info.get('side_1_coord_2'),
            sample_info.get('side_1_coord_3')
        )
        
        side2_coord = (
            sample_info.get('side_2_coord_1'),
            sample_info.get('side_2_coord_2'),
            sample_info.get('side_2_coord_3')
        )
        
        # Create segmented cube using the dataloader
        bbox_name = sample_info.get('bbox_name', '')
        cube = self.data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=config.subvol_size,
            alpha=config.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=True
        )
        
        # Extract the mask (regions not affected by gray overlay)
        # The mask is determined by where the RGB values are still the original values
        # In the overlaid_image calculation in create_segmented_cube:
        # overlaid_image = raw_rgb * mask_factor + (1 - mask_factor) * blended_part
        
        # To extract the mask from the cube, we need to:
        # 1. Transpose the cube back to match original dimensions (it's in y,x,c,z format)
        cube_transposed = np.transpose(cube, (3, 0, 1, 2))
        
        # 2. Check where the RGB channels are not affected by the gray overlay (equal values across channels)
        # For our binary mask, we'll consider a pixel as "masked" if it's affected by the overlay
        mask = np.zeros(cube_transposed.shape[:-1], dtype=bool)
        gray_epsilon = 1e-5  # Small threshold to account for floating point errors
        
        # Assuming RGB channels are equal in original pixels and different in gray overlay
        # This is a heuristic - if RGB channels are very close, it's likely not affected by gray overlay
        for z in range(cube_transposed.shape[0]):
            for y in range(cube_transposed.shape[1]):
                for x in range(cube_transposed.shape[2]):
                    r, g, b = cube_transposed[z, y, x, 0], cube_transposed[z, y, x, 1], cube_transposed[z, y, x, 2]
                    if abs(r - g) < gray_epsilon and abs(r - b) < gray_epsilon and abs(g - b) < gray_epsilon:
                        # Equal RGB values mean original pixel, which represents the "non-masked" region
                        mask[z, y, x] = True
        
        # Resize mask to match original shape if needed
        if mask.shape != original_shape:
            from scipy.ndimage import zoom
            zoom_factors = (
                original_shape[0] / mask.shape[0],
                original_shape[1] / mask.shape[1],
                original_shape[2] / mask.shape[2]
            )
            mask = zoom(mask.astype(float), zoom_factors, order=0).astype(bool)
        
        return mask
    
    def calculate_metrics(self, attention_maps, mask):
        """
        Calculate overlap metrics between attention maps and mask.
        
        Args:
            attention_maps: Dictionary of attention maps by layer
            mask: Binary mask
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for layer_name, attention_map in attention_maps.items():
            # Ensure attention map has same shape as mask
            if attention_map.shape != mask.shape:
                attention_resized = F.interpolate(
                    torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float(),
                    size=mask.shape,
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0).numpy()
            else:
                attention_resized = attention_map
            
            # Calculate metrics
            # 1. Dice coefficient (2 * intersection / sum of areas)
            attention_binary = attention_resized > 0.5  # Threshold the attention map
            intersection = np.logical_and(attention_binary, mask).sum()
            dice = (2 * intersection) / (attention_binary.sum() + mask.sum() + 1e-10)
            
            # 2. Intersection over Union (IoU)
            union = np.logical_or(attention_binary, mask).sum()
            iou = intersection / (union + 1e-10)
            
            # 3. Pearson correlation
            flat_attention = attention_resized.flatten()
            flat_mask = mask.astype(float).flatten()
            
            # Check if arrays have variation before calculating correlation
            if np.std(flat_attention) > 0 and np.std(flat_mask) > 0:
                correlation, _ = pearsonr(flat_attention, flat_mask)
            else:
                correlation = 0  # Default for constant arrays
            
            # 4. Mean attention in masked vs non-masked regions
            mean_attention_masked = attention_resized[mask].mean() if mask.any() else 0
            mean_attention_nonmasked = attention_resized[~mask].mean() if (~mask).any() else 0
            attention_ratio = mean_attention_masked / mean_attention_nonmasked if mean_attention_nonmasked > 0 else 0
            
            metrics[layer_name] = {
                'dice': dice,
                'iou': iou,
                'correlation': correlation,
                'mean_attention_masked': mean_attention_masked,
                'mean_attention_nonmasked': mean_attention_nonmasked,
                'attention_ratio': attention_ratio
            }
        
        return metrics
    
    def analyze_sample(self, sample_idx):
        """
        Analyze a single sample.
        
        Args:
            sample_idx: Index of the sample in the dataset
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing sample {sample_idx}...")
        
        # Get the sample
        pixel_values, syn_info, bbox_name = self.dataset[sample_idx]
        print(f"Sample shape: {pixel_values.shape}, BBox: {bbox_name}")
        
        # Convert to batch format and move to device
        input_tensor = pixel_values.float().unsqueeze(0).to(self.device)
        
        # Double-check dimensions - the model expects [B, C, D, H, W]
        if input_tensor.shape[1] != 1:  # If channels dimension is not 1
            # Probably in format [B, D, C, H, W], need to permute
            input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Get original image and dimensions
        original_img = pixel_values.cpu().numpy()
        original_shape = original_img.shape[0:3]  # D, H, W dimensions
        
        # Generate attention maps for all layers
        attention_maps = self._generate_attention_maps(input_tensor, syn_info)
        
        # Get the raw, segmentation, and additional mask volumes for this sample's bounding box
        if hasattr(self.dataset, 'vol_data_dict') and bbox_name in self.dataset.vol_data_dict:
            raw_vol, seg_vol, add_mask_vol = self.dataset.vol_data_dict[bbox_name]
            
            # Get the mask based on segmentation type
            mask = self._get_masks(raw_vol, seg_vol, add_mask_vol, syn_info, original_shape)
            
            # Calculate metrics
            metrics = self.calculate_metrics(attention_maps, mask)
            
            # Store results
            self.results[sample_idx] = {
                'bbox_name': bbox_name,
                'syn_info': syn_info,
                'metrics': metrics,
                'attention_maps': attention_maps,
                'mask': mask,
                'original_img': original_img
            }
            
            return self.results[sample_idx]
        else:
            print(f"Error: Could not find volume data for bounding box {bbox_name}")
            return None
    
    def analyze_multiple_samples(self, sample_indices):
        """
        Analyze multiple samples.
        
        Args:
            sample_indices: List of sample indices to analyze
            
        Returns:
            Dictionary with analysis results for all samples
        """
        for sample_idx in sample_indices:
            if sample_idx >= len(self.dataset):
                print(f"Warning: Sample index {sample_idx} exceeds dataset size ({len(self.dataset)}). Skipping.")
                continue
            
            self.analyze_sample(sample_idx)
        
        return self.results
    
    def visualize_sample(self, sample_idx, n_slices=4):
        """
        Visualize attention maps for a sample.
        
        Args:
            sample_idx: Index of the sample to visualize
            n_slices: Number of slices to visualize
            
        Returns:
            Path to the saved visualization
        """
        if sample_idx not in self.results:
            print(f"Sample {sample_idx} not analyzed yet. Running analysis...")
            self.analyze_sample(sample_idx)
        
        result = self.results[sample_idx]
        original_img = result['original_img']
        mask = result['mask']
        attention_maps = result['attention_maps']
        bbox_name = result['bbox_name']
        
        # Debug: Print shapes for debugging
        print(f"Debug - Original image shape: {original_img.shape}")
        print(f"Debug - Mask shape: {mask.shape}")
        for layer_name in self.layers:
            print(f"Debug - Attention map shape for {layer_name}: {attention_maps[layer_name].shape}")
        
        # Fix mask shape if needed (add missing width dimension)
        if len(mask.shape) == 3 and mask.shape[2] != original_img.shape[3]:
            print(f"Fixing mask shape from {mask.shape} to match original image width")
            # Assuming mask is (depth, channels, height) and missing width
            # Reshape to (depth, channels, height, 1) and repeat to match width
            mask = np.repeat(mask[:, :, :, np.newaxis], original_img.shape[3], axis=3)
            print(f"New mask shape: {mask.shape}")
        
        # Select slices to visualize
        depth = original_img.shape[0]
        slice_indices = np.linspace(0, depth-1, n_slices, dtype=int)
        
        # Create figure
        fig, axes = plt.subplots(n_slices, len(self.layers) + 2, figsize=(3 * (len(self.layers) + 2), 3 * n_slices))
        plt.suptitle(f"Attention Map Analysis - Sample {sample_idx} (Bbox: {bbox_name})", fontsize=16)
        
        # Modify colormap for better visibility of weak signals
        # Create a custom colormap that goes from transparent to red with better visibility for weak signals
        colors = [(1, 1, 1, 0),          # Transparent white for very low values
                 (1, 0.9, 0.9, 0.2),     # Very light pink with some transparency for low values
                 (1, 0.7, 0.7, 0.4),     # Light pink-red with more opacity for low-mid values
                 (1, 0.5, 0.5, 0.6),     # Stronger pink-red with higher opacity for mid values
                 (1, 0.3, 0.3, 0.8),     # Intense red with high opacity for high values
                 (1, 0, 0, 1)]           # Pure red for highest values
        attention_cmap = LinearSegmentedColormap.from_list('custom_red', colors, N=256)
        
        # Add column titles
        axes[0, 0].set_title("Original", fontsize=12)
        axes[0, 1].set_title("Mask", fontsize=12)
        
        for col, layer_name in enumerate(self.layers, 2):
            axes[0, col].set_title(f"{self.layer_name_map[layer_name]}", fontsize=12)
        
        # Add row titles (slice numbers)
        for slice_idx, slice_num in enumerate(slice_indices):
            axes[slice_idx, 0].set_ylabel(f"Slice {slice_num}", fontsize=12)
        
        # Plot original images and attention maps
        for slice_idx, slice_num in enumerate(slice_indices):
            # Original image
            axes[slice_idx, 0].imshow(original_img[slice_num, 0], cmap='gray', vmin=0, vmax=1)
            axes[slice_idx, 0].axis('off')
            
            # Mask
            mask_slice = mask[slice_num, 0] if len(mask.shape) == 4 else mask[slice_num]
            axes[slice_idx, 1].imshow(original_img[slice_num, 0], cmap='gray', vmin=0, vmax=1)
            mask_overlay = np.zeros((*mask_slice.shape, 4))
            mask_overlay[mask_slice > 0] = [0, 1, 0, 0.5]  # Green with 50% opacity
            axes[slice_idx, 1].imshow(mask_overlay)
            axes[slice_idx, 1].axis('off')
            
            # Attention maps for each layer
            for col, layer_name in enumerate(self.layers, 2):
                attention = attention_maps[layer_name]
                
                # Resize attention to match original image dimensions (without channels)
                target_shape = (original_img.shape[0], original_img.shape[2], original_img.shape[3])
                
                if attention.shape != target_shape:
                    print(f"Resizing attention map for layer {layer_name} from {attention.shape} to {target_shape}")
                    
                    # Convert to tensor and add batch/channel dimensions
                    attention_tensor = torch.from_numpy(attention).float()
                    if len(attention_tensor.shape) == 3:  # Already 3D
                        attention_tensor = attention_tensor.unsqueeze(0).unsqueeze(0)
                    else:  # Lower dimension, need to reshape
                        print(f"Warning: Unexpected shape for attention map: {attention.shape}")
                        continue  # Skip this layer if shape is unexpected
                    
                    # Resize to target shape
                    try:
                        resized_tensor = F.interpolate(
                            attention_tensor,
                            size=target_shape,
                            mode='trilinear',
                            align_corners=False
                        )
                        attention_resized = resized_tensor.squeeze(0).squeeze(0).numpy()
                    except Exception as e:
                        print(f"Error resizing attention map: {e}")
                        continue  # Skip this layer if resize fails
                else:
                    attention_resized = attention
                
                # Display original image
                axes[slice_idx, col].imshow(original_img[slice_num, 0], cmap='gray', vmin=0, vmax=1)
                
                # Overlay attention map
                try:
                    attention_slice = attention_resized[slice_num]
                    
                    # Apply logarithmic transformation to enhance visibility of weak signals
                    # Adding a small epsilon to avoid log(0) and scaling the result
                    epsilon = 1e-5
                    attention_enhanced = np.log1p(attention_slice * 10) / np.log1p(10)
                    
                    # Ensure values are within [0, 1] range
                    if attention_enhanced.max() > attention_enhanced.min():
                        attention_enhanced = (attention_enhanced - attention_enhanced.min()) / (attention_enhanced.max() - attention_enhanced.min() + epsilon)
                    
                    im = axes[slice_idx, col].imshow(attention_enhanced, cmap=attention_cmap, alpha=0.7, vmin=0, vmax=1)
                except Exception as e:
                    print(f"Error displaying attention map for layer {layer_name}, slice {slice_num}: {e}")
                    continue  # Skip this slice if display fails
                
                # Highlight mask boundary
                try:
                    from scipy.ndimage import binary_erosion
                    mask_boundary = mask_slice > 0
                    if np.any(mask_boundary):
                        eroded = binary_erosion(mask_boundary, iterations=1)
                        boundary = mask_boundary & ~eroded
                        
                        # Only draw contour if the boundary has enough points
                        if boundary.shape[0] >= 2 and boundary.shape[1] >= 2 and np.any(boundary):
                            axes[slice_idx, col].contour(boundary, colors=['blue'], linewidths=0.5)
                except Exception as e:
                    print(f"Warning: Could not draw contour for layer {layer_name}, slice {slice_num}: {e}")
                
                axes[slice_idx, col].axis('off')
                
                # Add colorbar to last row only
                if slice_idx == n_slices - 1 and col == len(self.layers) + 1:
                    divider = make_axes_locatable(axes[slice_idx, col])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        output_file = os.path.join(self.output_dir, f"attention_mask_overlap_sample_{sample_idx}_bbox_{bbox_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_file}")
        return output_file
    
    def generate_report(self):
        """
        Generate a statistical report of the analysis results.
        
        Returns:
            DataFrame with analysis results
        """
        if not self.results:
            print("No results to report. Please analyze some samples first.")
            return None
        
        # Create a list to store results
        report_data = []
        
        for sample_idx, result in self.results.items():
            bbox_name = result['bbox_name']
            synapse_id = result['syn_info'].get('Var1', f'Sample_{sample_idx}')
            
            for layer_name in self.layers:
                metrics = result['metrics'][layer_name]
                
                # Add row to report
                report_data.append({
                    'sample_idx': sample_idx,
                    'bbox_name': bbox_name,
                    'synapse_id': synapse_id,
                    'layer_name': layer_name,
                    'layer_description': self.layer_name_map[layer_name],
                    'dice_coefficient': metrics['dice'],
                    'iou': metrics['iou'],
                    'correlation': metrics['correlation'],
                    'mean_attention_masked': metrics['mean_attention_masked'],
                    'mean_attention_nonmasked': metrics['mean_attention_nonmasked'],
                    'attention_ratio': metrics['attention_ratio'],
                    'segmentation_type': self.segmentation_type
                })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Save report to CSV
        report_path = os.path.join(self.output_dir, f"attention_mask_analysis_seg{self.segmentation_type}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Report saved to {report_path}")
        
        # Generate summary statistics by layer
        summary = report_df.groupby('layer_name').agg({
            'dice_coefficient': ['mean', 'std', 'min', 'max'],
            'iou': ['mean', 'std', 'min', 'max'],
            'correlation': ['mean', 'std', 'min', 'max'],
            'attention_ratio': ['mean', 'std', 'min', 'max']
        })
        
        # Save summary to CSV
        summary_path = os.path.join(self.output_dir, f"attention_mask_summary_seg{self.segmentation_type}.csv")
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to {summary_path}")
        
        # Create and save visualization of mean metrics by layer
        self._plot_layer_metrics(report_df)
        
        return report_df
    
    def _plot_layer_metrics(self, report_df):
        """
        Plot metrics by layer.
        
        Args:
            report_df: DataFrame with results
        """
        metrics = ['dice_coefficient', 'iou', 'correlation', 'attention_ratio']
        metric_names = ['Dice Coefficient', 'IoU', 'Correlation', 'Attention Ratio (Masked/Non-masked)']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            # Calculate mean and std by layer
            grouped = report_df.groupby('layer_description')[metric].agg(['mean', 'std'])
            
            # Sort by layer order
            sorted_indices = [self.layer_name_map.get(layer) for layer in self.layers]
            sorted_grouped = grouped.loc[sorted_indices]
            
            # Plot
            ax = axes[i]
            x = np.arange(len(sorted_grouped))
            ax.bar(x, sorted_grouped['mean'], yerr=sorted_grouped['std'], capsize=5, 
                  color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(name)
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_grouped.index, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for j, v in enumerate(sorted_grouped['mean']):
                ax.text(j, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"layer_metrics_comparison_seg{self.segmentation_type}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Layer metrics comparison plot saved to {plot_path}")


def main():
    """
    Main function to run the attention-mask analysis.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze overlap between attention maps and masks')
    parser.add_argument('--sample_idx', type=int, default=0,
                      help='Sample index to analyze')
    parser.add_argument('--output_dir', type=str, default='results/attention_mask_analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--n_slices', type=int, default=4,
                      help='Number of slices to visualize')
    parser.add_argument('--batch_mode', action='store_true',
                      help='Process multiple samples in batch mode')
    parser.add_argument('--n_samples', type=int, default=5,
                      help='Number of random samples to process in batch mode')
    parser.add_argument('--specific_indices', type=int, nargs='+',
                      help='Process specific sample indices instead of random ones')
    parser.add_argument('--segmentation_type', type=int, default=5,
                      help='Segmentation type to use (default: 5)')
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
    
    # Set normalization settings
    if hasattr(processor, 'normalize_volume'):
        processor.normalize_volume = True
        print("Set processor.normalize_volume = True")
    
    # Use the segmentation type specified in args instead of config
    segmentation_type = args.segmentation_type
    print(f"Using segmentation type: {segmentation_type}")
    
    # Create dataset with explicit volume-wide normalization
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=segmentation_type,
        alpha=config.alpha,
        normalize_across_volume=True
    )
    print(f"Created dataset with {len(dataset)} samples")
    
    # Initialize analyzer
    analyzer = AttentionMaskAnalyzer(
        model=model,
        dataset=dataset,
        segmentation_type=segmentation_type,  # Use the specified segmentation type
        output_dir=f"{args.output_dir}_seg{segmentation_type}"  # Include segmentation type in the output dir
    )
    
    if args.batch_mode:
        # Process multiple samples
        if args.specific_indices:
            sample_indices = args.specific_indices
            print(f"Using specified indices: {sample_indices}")
        else:
            # Generate random sample indices
            sample_indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
            print(f"Generated random sample indices: {sample_indices}")
        
        # Analyze and visualize samples
        analyzer.analyze_multiple_samples(sample_indices)
        for sample_idx in sample_indices:
            analyzer.visualize_sample(sample_idx, n_slices=args.n_slices)
        
        # Generate report
        analyzer.generate_report()
    else:
        # Process single sample
        analyzer.analyze_sample(args.sample_idx)
        analyzer.visualize_sample(args.sample_idx, n_slices=args.n_slices)
        analyzer.generate_report()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 