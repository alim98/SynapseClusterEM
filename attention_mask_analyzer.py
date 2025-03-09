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
from scipy.stats import pearsonr, ttest_ind, f_oneway
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from synapse import config, SynapseDataset, Synapse3DProcessor
from synapse.data.dataloader import SynapseDataLoader
from inference import VGG3D, load_and_prepare_data
# Import classes from multi_layer_cam.py
from multi_layer_cam import SimpleGradCAM, normalize_globally, process_single_sample


class AttentionMaskAnalyzer:
    """
    Analyzes the overlap between attention maps and masked regions in 3D electron microscopy data.
    """
    
    def __init__(self, model, dataset, segmentation_type, output_dir='results/attention_analysis', use_cuda=True, batch_size=5):
        """
        Initialize the analyzer.
        
        Args:
            model: The VGG3D model
            dataset: The SynapseDataset
            segmentation_type: Segmentation type to use for mask generation
            output_dir: Directory to save analysis results
            use_cuda: Whether to use CUDA for running the model
            batch_size: Number of samples to process before clearing cache
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
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
    
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
        Extract masks based on segmentation type directly from segmentation volumes.
        
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
        
        # Extract bounding box information for label identification
        bbox_name = sample_info.get('bbox_name', '')
        bbox_num = bbox_name.replace("bbox", "").strip()
        
        # Determine label values based on bbox
        if bbox_num in {'2', '5',}:
            mito_label = 1
            vesicle_label = 3
            cleft_label2 = 4
            cleft_label = 2
        elif bbox_num == '7':
            mito_label = 1
            vesicle_label = 2
            cleft_label2 = 3
            cleft_label = 4
        elif bbox_num == '4':
            mito_label = 3
            vesicle_label = 2
            cleft_label2 = 4
            cleft_label = 1
        elif bbox_num == '3':
            mito_label = 6
            vesicle_label = 7
            cleft_label2 = 8
            cleft_label = 9
        else:
            mito_label = 5
            vesicle_label = 6
            cleft_label = 7
            cleft_label2 = 7
        
        # Calculate crop boundaries (80x80x80 centered on synapse)
        half_size = config.subvol_size // 2
        cx, cy, cz = central_coord
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, raw_vol.shape[2])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, raw_vol.shape[1])
        z_start = max(cz - half_size, 0)
        z_end = min(cz + half_size, raw_vol.shape[0])
        
        # Get vesicle mask
        vesicle_full_mask = (add_mask_vol == vesicle_label)
        vesicle_mask = self.data_loader.get_closest_component_mask(
            vesicle_full_mask,
            z_start, z_end,
            y_start, y_end,
            x_start, x_end,
            (cx, cy, cz)
        )
        
        # Create segment masks for the two sides
        def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
            x1, y1, z1 = s1_coord
            x2, y2, z2 = s2_coord
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            return mask_1, mask_2
        
        mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)
        
        # Determine which side is the presynaptic side
        overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
        
        # Create mask based on segmentation type (directly, not from overlay)
        if self.segmentation_type == 0:
            combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
        elif self.segmentation_type == 1:
            combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        elif self.segmentation_type == 2:
            combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
        elif self.segmentation_type == 3:
            combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
        elif self.segmentation_type == 4:
            vesicle_closest = self.data_loader.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest = self.data_loader.get_closest_component_mask(
                ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest2 = self.data_loader.get_closest_component_mask(
                ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest, cleft_closest2))
        elif self.segmentation_type == 5:
            vesicle_closest = self.data_loader.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest = self.data_loader.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_extra = np.logical_or(vesicle_closest, cleft_closest)
            combined_mask_full = np.logical_or(mask_1_full, np.logical_or(mask_2_full, combined_mask_extra))
        elif self.segmentation_type == 6:
            combined_mask_full = self.data_loader.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
        elif self.segmentation_type == 7:
            cleft_closest = self.data_loader.get_closest_component_mask(
                ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest2 = self.data_loader.get_closest_component_mask(
                ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_full = np.logical_or(cleft_closest, cleft_closest2)
        elif self.segmentation_type == 8:
            combined_mask_full = self.data_loader.get_closest_component_mask(
                (add_mask_vol == mito_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
        elif self.segmentation_type == 10:
            cleft_closest = self.data_loader.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
            combined_mask_full = np.logical_or(cleft_closest, pre_mask_full)
        elif self.segmentation_type == 9:
            vesicle_closest = self.data_loader.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest = self.data_loader.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_full = np.logical_or(cleft_closest, vesicle_closest)
        else:
            raise ValueError(f"Unsupported segmentation type: {self.segmentation_type}")
        
        # Extract the subvolume from the mask
        sub_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary to match the desired subvolume size
        pad_z = config.subvol_size - sub_mask.shape[0]
        pad_y = config.subvol_size - sub_mask.shape[1]
        pad_x = config.subvol_size - sub_mask.shape[2]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            sub_mask = np.pad(sub_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)
        
        # Ensure the mask has the exact size expected
        sub_mask = sub_mask[:config.subvol_size, :config.subvol_size, :config.subvol_size]
        
        # Resize mask to match original shape if needed
        if sub_mask.shape != original_shape:
            from scipy.ndimage import zoom
            zoom_factors = (
                original_shape[0] / sub_mask.shape[0],
                original_shape[1] / sub_mask.shape[1],
                original_shape[2] / sub_mask.shape[2]
            )
            sub_mask = zoom(sub_mask.astype(float), zoom_factors, order=0).astype(bool)
        
        print(f"Generated mask for segmentation type {self.segmentation_type}, shape: {sub_mask.shape}")
        
        return sub_mask
    
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
        
        # Make sure mask is properly shaped
        # The mask should be 3D (depth, height, width) to match original image
        if len(mask.shape) == 3 and mask.shape[0] == original_img.shape[0]:
            print("Mask is already properly shaped for visualization")
        elif len(mask.shape) == 2:
            # If mask is 2D, expand it to match the depth dimension
            print(f"Expanding 2D mask to 3D to match image depth")
            mask = np.expand_dims(mask, axis=0)
            mask = np.repeat(mask, original_img.shape[0], axis=0)
        # If mask is missing dimensions or has wrong shape
        elif len(mask.shape) < 3:
            print(f"Warning: Unexpected mask shape: {mask.shape}. Trying to reshape...")
            # Reshape to match original image shape
            try:
                mask = mask.reshape(original_img.shape[0], original_img.shape[2], original_img.shape[3])
                print(f"Successfully reshaped mask to {mask.shape}")
            except Exception as e:
                print(f"Error reshaping mask: {e}")
                # Create an empty mask as fallback
                mask = np.zeros((original_img.shape[0], original_img.shape[2], original_img.shape[3]), dtype=bool)
        
        # Debug: Print final mask shape after adjustments
        print(f"Final mask shape for visualization: {mask.shape}")
        
        # Fix mask shape if needed (add missing width dimension)
        if len(mask.shape) == 3 and mask.shape[2] != original_img.shape[3]:
            print(f"Fixing mask shape from {mask.shape} to match original image width")
            # Assuming mask is (depth, height, width) or needs to be resized
            from scipy.ndimage import zoom
            target_shape = (original_img.shape[0], original_img.shape[2], original_img.shape[3])
            zoom_factors = [target_shape[i] / mask.shape[i] for i in range(len(mask.shape))]
            mask = zoom(mask.astype(float), zoom_factors, order=0).astype(bool)
            print(f"New mask shape after resizing: {mask.shape}")
        
        # Select slices to visualize
        depth = original_img.shape[0]
        slice_indices = np.linspace(0, depth-1, n_slices, dtype=int)
        
        # Create figure - REMOVING mask column, now only original and attention maps
        fig, axes = plt.subplots(n_slices, len(self.layers) + 1, figsize=(3 * (len(self.layers) + 1), 3 * n_slices))
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
        
        # Add column titles - no mask column
        axes[0, 0].set_title("Original", fontsize=12)
        
        for col, layer_name in enumerate(self.layers, 1):  # Start from 1 since mask column is removed
            axes[0, col].set_title(f"{self.layer_name_map[layer_name]}", fontsize=12)
        
        # Add row titles (slice numbers)
        for slice_idx, slice_num in enumerate(slice_indices):
            axes[slice_idx, 0].set_ylabel(f"Slice {slice_num}", fontsize=12)
        
        # Plot original images and attention maps
        for slice_idx, slice_num in enumerate(slice_indices):
            # Original image
            axes[slice_idx, 0].imshow(original_img[slice_num, 0], cmap='gray', vmin=0, vmax=1)
            axes[slice_idx, 0].axis('off')
            
            # Attention maps for each layer - adjusted column indices since mask is removed
            for col, layer_name in enumerate(self.layers, 1):  # Start from 1 since mask column is removed
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
                
                axes[slice_idx, col].axis('off')
                
                # Add colorbar to last row only
                if slice_idx == n_slices - 1 and col == len(self.layers):
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
        
        # Extract all metrics
        for sample_idx, sample_result in self.results.items():
            bbox_name = sample_result['bbox_name']
            metrics = sample_result['metrics']
            
            for layer_name, layer_metrics in metrics.items():
                row = {
                    'sample_idx': sample_idx,
                    'bbox_name': bbox_name,
                    'layer': layer_name,
                    'layer_name': self.layer_name_map[layer_name]
                }
                
                # Add all metrics
                for metric_name, metric_value in layer_metrics.items():
                    row[metric_name] = metric_value
                
                report_data.append(row)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Save report to CSV
        report_path = os.path.join(self.output_dir, f"attention_mask_analysis_seg{self.segmentation_type}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Report saved to {report_path}")
        
        # Create summary statistics by layer
        summary_df = report_df.groupby('layer').agg({
            'dice': ['mean', 'std', 'min', 'max', 'median'],
            'iou': ['mean', 'std', 'min', 'max', 'median'],
            'correlation': ['mean', 'std', 'min', 'max', 'median'],
            'mean_attention_masked': ['mean', 'std', 'min', 'max', 'median'],
            'mean_attention_nonmasked': ['mean', 'std', 'min', 'max', 'median'],
            'attention_ratio': ['mean', 'std', 'min', 'max', 'median']
        })
        
        # Calculate confidence intervals (95%)
        group_sizes = report_df.groupby('layer').size()
        confidence_intervals = {}
        
        for layer in summary_df.index:
            confidence_intervals[layer] = {}
            for metric in ['dice', 'iou', 'correlation', 'attention_ratio']:
                values = report_df[report_df['layer'] == layer][metric]
                n = len(values)
                if n > 1:
                    # Calculate 95% confidence interval
                    mean = values.mean()
                    sem = values.std() / np.sqrt(n)
                    ci_95 = (mean - 1.96 * sem, mean + 1.96 * sem)
                    confidence_intervals[layer][f"{metric}_ci_lower"] = ci_95[0]
                    confidence_intervals[layer][f"{metric}_ci_upper"] = ci_95[1]
                else:
                    confidence_intervals[layer][f"{metric}_ci_lower"] = np.nan
                    confidence_intervals[layer][f"{metric}_ci_upper"] = np.nan
        
        # Add confidence intervals to summary
        ci_df = pd.DataFrame(confidence_intervals).T
        summary_df = pd.concat([summary_df, ci_df], axis=1)
        
        # Save summary statistics to CSV
        summary_path = os.path.join(self.output_dir, f"attention_mask_summary_seg{self.segmentation_type}.csv")
        summary_df.to_csv(summary_path)
        print(f"Summary statistics saved to {summary_path}")
        
        # Create plots for better visualization
        self._plot_layer_metrics(report_df)
        self._plot_boxplots(report_df)
        self._plot_distributions(report_df)
        self._plot_correlation_heatmap(report_df)
        self._perform_statistical_tests(report_df)
        self._plot_roc_curves(report_df)
        
        return report_df
    
    def _plot_layer_metrics(self, report_df):
        """
        Create bar plots comparing metrics across layers.
        
        Args:
            report_df: DataFrame with analysis results
        """
        # Group by layer
        grouped = report_df.groupby('layer').agg({
            'dice': ['mean', 'std'],
            'iou': ['mean', 'std'],
            'correlation': ['mean', 'std'],
            'attention_ratio': ['mean', 'std']
        })
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metrics = ['dice', 'iou', 'correlation', 'attention_ratio']
        titles = ['Dice Coefficient', 'IoU', 'Correlation', 'Attention Ratio (Masked/Non-masked)']
        
        for i, (name, ax) in enumerate(zip(metrics, axes)):
            # Get data for this metric
            sorted_grouped = grouped[name].sort_values('mean', ascending=False)
            
            # Create x positions
            x = np.arange(len(sorted_grouped.index))
            
            # Create bar plot
            ax.bar(x, sorted_grouped['mean'], yerr=sorted_grouped['std'], capsize=5, 
                  color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(titles[i])
            ax.set_xticks(x)
            ax.set_xticklabels([self.layer_name_map[l] for l in sorted_grouped.index], rotation=45, ha='right')
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
    
    def _plot_boxplots(self, report_df):
        """
        Create boxplots to show distribution of metrics across layers.
        
        Args:
            report_df: DataFrame with analysis results
        """
        # Map layer names for better readability
        report_df_plot = report_df.copy()
        report_df_plot['Layer'] = report_df_plot['layer'].map(self.layer_name_map)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metrics = ['dice', 'iou', 'correlation', 'attention_ratio']
        titles = ['Dice Coefficient', 'IoU', 'Correlation', 'Attention Ratio (Masked/Non-masked)']
        
        for i, (metric, title, ax) in enumerate(zip(metrics, titles, axes)):
            # Create boxplot
            sns.boxplot(x='Layer', y=metric, data=report_df_plot, ax=ax, palette='Set3')
            ax.set_title(title)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add sample size
            sample_sizes = report_df_plot.groupby('Layer').size()
            for j, layer in enumerate(ax.get_xticklabels()):
                ax.text(j, ax.get_ylim()[0], f"n={sample_sizes[layer.get_text()]}", 
                       ha='center', va='top', fontsize=8, rotation=0, color='blue')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"layer_boxplots_seg{self.segmentation_type}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Layer boxplots saved to {plot_path}")
    
    def _plot_distributions(self, report_df):
        """
        Create distribution plots to visualize the distribution of metrics across layers.
        
        Args:
            report_df: DataFrame with analysis results
        """
        # Map layer names for better readability
        report_df_plot = report_df.copy()
        report_df_plot['Layer'] = report_df_plot['layer'].map(self.layer_name_map)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Plot each metric
        metrics = ['dice', 'iou', 'correlation', 'attention_ratio']
        titles = ['Dice Coefficient Distribution', 'IoU Distribution', 
                 'Correlation Distribution', 'Attention Ratio Distribution']
        
        for i, (metric, title, ax) in enumerate(zip(metrics, titles, axes)):
            # Create KDE plot for each layer
            for layer_name in self.layer_names:
                layer_data = report_df_plot[report_df_plot['Layer'] == layer_name][metric]
                # Only plot if we have enough data points
                if len(layer_data) > 1:
                    sns.kdeplot(layer_data, ax=ax, label=layer_name)
            
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"metric_distributions_seg{self.segmentation_type}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metric distribution plots saved to {plot_path}")
    
    def _plot_correlation_heatmap(self, report_df):
        """
        Create heatmap to visualize correlations between different metrics.
        
        Args:
            report_df: DataFrame with analysis results
        """
        # Calculate correlation matrix for each layer
        for layer in self.layers:
            layer_df = report_df[report_df['layer'] == layer]
            if len(layer_df) < 2:
                continue  # Skip if not enough data
                
            metrics = ['dice', 'iou', 'correlation', 'mean_attention_masked', 
                       'mean_attention_nonmasked', 'attention_ratio']
            
            # Calculate correlation matrix
            corr_matrix = layer_df[metrics].corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                       linewidths=.5, fmt='.2f')
            plt.title(f"Metric Correlations for {self.layer_name_map[layer]}")
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f"correlation_heatmap_{layer.replace('.', '_')}_seg{self.segmentation_type}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Correlation heatmap for {layer} saved to {plot_path}")
    
    def _perform_statistical_tests(self, report_df):
        """
        Perform statistical tests to compare metrics across layers.
        
        Args:
            report_df: DataFrame with analysis results
        """
        # Create a DataFrame for statistics
        stats_data = []
        
        # Metrics to test
        metrics = ['dice', 'iou', 'correlation', 'attention_ratio']
        metric_names = ['Dice Coefficient', 'IoU', 'Correlation', 'Attention Ratio']
        
        # ANOVA test for each metric
        for metric, metric_name in zip(metrics, metric_names):
            # Group data by layer
            groups = [report_df[report_df['layer'] == layer][metric].dropna() for layer in self.layers]
            
            # Only perform ANOVA if at least 2 groups have enough data
            valid_groups = [g for g in groups if len(g) > 1]
            if len(valid_groups) >= 2:
                f_val, p_val = f_oneway(*valid_groups)
                
                stats_data.append({
                    'Test': 'ANOVA',
                    'Metric': metric_name,
                    'Statistic': f_val,
                    'p-value': p_val,
                    'Significant': p_val < 0.05
                })
                
                # Perform pairwise t-tests between layers
                for i, layer1 in enumerate(self.layers):
                    for j, layer2 in enumerate(self.layers):
                        if i < j:  # Only compare each pair once
                            data1 = report_df[report_df['layer'] == layer1][metric].dropna()
                            data2 = report_df[report_df['layer'] == layer2][metric].dropna()
                            
                            # Only perform t-test if both groups have enough data
                            if len(data1) > 1 and len(data2) > 1:
                                t_val, p_val = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
                                
                                stats_data.append({
                                    'Test': 't-test',
                                    'Metric': metric_name,
                                    'Comparison': f"{self.layer_name_map[layer1]} vs {self.layer_name_map[layer2]}",
                                    'Statistic': t_val,
                                    'p-value': p_val,
                                    'Significant': p_val < 0.05
                                })
        
        # Create DataFrame with statistics
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            # Save to CSV
            stats_path = os.path.join(self.output_dir, f"statistical_tests_seg{self.segmentation_type}.csv")
            stats_df.to_csv(stats_path, index=False)
            print(f"Statistical test results saved to {stats_path}")
            
            # Create summary plot for ANOVA p-values
            anova_stats = stats_df[stats_df['Test'] == 'ANOVA']
            if len(anova_stats) > 0:
                plt.figure(figsize=(10, 6))
                
                # Create barplot of p-values
                bars = plt.bar(anova_stats['Metric'], anova_stats['p-value'], color='skyblue')
                
                # Mark significant results
                for i, bar in enumerate(bars):
                    if anova_stats.iloc[i]['Significant']:
                        bar.set_color('green')
                        bar.set_alpha(0.7)
                
                plt.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
                plt.ylabel('p-value')
                plt.title('ANOVA Test Results Across Metrics')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(self.output_dir, f"anova_results_seg{self.segmentation_type}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"ANOVA test results plot saved to {plot_path}")
    
    def _plot_roc_curves(self, report_df):
        """
        Plot ROC curves for attention maps per layer.
        This treats the masks as ground truth and evaluates 
        how well attention maps can discriminate masked from non-masked regions.
        
        Args:
            report_df: DataFrame with analysis results
        """
        # We need sample-level data for this, so iterate through results
        layer_auc_data = []
        
        for sample_idx, sample_result in self.results.items():
            # Get original shape
            if 'original_img' not in sample_result or 'mask' not in sample_result:
                continue
                
            mask = sample_result['mask'].flatten()
            attention_maps = sample_result['attention_maps']
            
            plt.figure(figsize=(10, 8))
            
            for layer_name in self.layers:
                if layer_name not in attention_maps:
                    continue
                    
                # Resize attention map to match mask if needed
                attention_map = attention_maps[layer_name]
                if attention_map.shape != sample_result['mask'].shape:
                    attention_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float()
                    resized_tensor = F.interpolate(
                        attention_tensor,
                        size=sample_result['mask'].shape,
                        mode='trilinear',
                        align_corners=False
                    )
                    attention_map = resized_tensor.squeeze(0).squeeze(0).numpy()
                
                # Flatten for ROC calculation
                attention_flat = attention_map.flatten()
                
                # Only calculate if we have both positive and negative samples
                if np.sum(mask) > 0 and np.sum(mask) < len(mask):
                    # Calculate ROC
                    fpr, tpr, _ = roc_curve(mask, attention_flat)
                    roc_auc = auc(fpr, tpr)
                    
                    # Store AUC data
                    layer_auc_data.append({
                        'sample_idx': sample_idx,
                        'layer': layer_name,
                        'auc': roc_auc
                    })
                    
                    # Plot ROC curve for this sample
                    plt.plot(fpr, tpr, lw=2, alpha=0.7,
                            label=f'{self.layer_name_map[layer_name]} (AUC = {roc_auc:.2f})')
            
            # Finalize plot
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for Sample {sample_idx}')
            plt.legend(loc="lower right")
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f"roc_curves_sample_{sample_idx}_seg{self.segmentation_type}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Summarize AUC values across samples
        if layer_auc_data:
            auc_df = pd.DataFrame(layer_auc_data)
            
            # Create boxplot of AUC values by layer
            plt.figure(figsize=(10, 6))
            auc_df['Layer'] = auc_df['layer'].map(self.layer_name_map)
            sns.boxplot(x='Layer', y='auc', data=auc_df, palette='Set3')
            plt.title('AUC Distribution by Layer')
            plt.xlabel('Layer')
            plt.ylabel('Area Under ROC Curve')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f"auc_distribution_seg{self.segmentation_type}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"AUC distribution plot saved to {plot_path}")
            
            # Save AUC data
            auc_path = os.path.join(self.output_dir, f"auc_values_seg{self.segmentation_type}.csv")
            auc_df.to_csv(auc_path, index=False)
            print(f"AUC values saved to {auc_path}")
            
            # Calculate average AUC by layer
            auc_summary = auc_df.groupby('layer').agg({
                'auc': ['mean', 'std', 'min', 'max', 'count']
            })
            
            # Report the best layer based on AUC
            best_layer = auc_summary['auc'].sort_values('mean', ascending=False).index[0]
            print(f"\nBased on AUC analysis, the best performing layer is: {self.layer_name_map[best_layer]}")
            print(f"Average AUC: {auc_summary.loc[best_layer, ('auc', 'mean')]:.4f}")
            
            return auc_summary


def main():
    """
    Main entry point for the attention mask analyzer script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze overlap between attention maps and segmentation masks')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to analyze (if not in batch mode)')
    parser.add_argument('--output_dir', type=str, default='results/attention_mask_analysis',
                        help='Directory to save output')
    parser.add_argument('--n_slices', type=int, default=4,
                        help='Number of slices to visualize')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process multiple samples in batch mode')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to analyze in batch mode')
    parser.add_argument('--n_visualize', type=int, default=10,
                        help='Number of random samples to visualize (when analyzing all)')
    parser.add_argument('--specific_indices', type=int, nargs='+',
                        help='Specific sample indices to analyze')
    parser.add_argument('--segmentation_type', type=int, default=5,
                        help='Type of segmentation mask to use')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Skip analysis and only visualize previously analyzed samples')
    parser.add_argument('--analyze_all', action='store_true',
                        help='Analyze the entire dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up
    from synapse import config
    from synapse.data import SynapseDataset
    from inference import VGG3D, load_and_prepare_data
    from synapse import Synapse3DProcessor
    
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
    
    # Use the segmentation type specified in args
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
    
    # Create analyzer
    print(f"Creating analyzer with segmentation type {segmentation_type}...")
    analyzer = AttentionMaskAnalyzer(
        model=model,
        dataset=dataset,
        segmentation_type=segmentation_type,
        output_dir=f"{args.output_dir}_seg{segmentation_type}", # Remove the extra space
    )
    
    if args.analyze_all:
        # Analyze the entire dataset
        print(f"Analyzing all {len(dataset)} samples...")
        
        # Process the entire dataset
        all_indices = list(range(len(dataset)))
        
        if not args.visualize_only:
            # Analyze all samples
            print("Running analysis on the entire dataset...")
            analyzer.analyze_multiple_samples(all_indices)
            
            # Generate report on all samples
            print("Generating report for all samples...")
            analyzer.generate_report()
        
        # Visualize only a random subset
        print(f"Visualizing {args.n_visualize} random samples...")
        if args.specific_indices and len(args.specific_indices) > 0:
            # Use specified indices for visualization if provided
            viz_indices = args.specific_indices[:args.n_visualize]
        else:
            # Otherwise, choose random indices
            viz_indices = np.random.choice(all_indices, min(args.n_visualize, len(dataset)), replace=False)
        
        print(f"Selected indices for visualization: {viz_indices}")
        for idx in viz_indices:
            analyzer.visualize_sample(idx, n_slices=args.n_slices)
    
    elif args.batch_mode:
        # Get specific sample indices or generate random ones
        if args.specific_indices:
            sample_indices = args.specific_indices
            print(f"Using specified sample indices: {sample_indices}")
        else:
            sample_indices = np.random.choice(len(dataset), min(args.n_samples, len(dataset)), replace=False)
            print(f"Generated random sample indices: {sample_indices}")
        
        # Analyze and visualize samples
        if not args.visualize_only:
            print("Running analysis on multiple samples...")
            analyzer.analyze_multiple_samples(sample_indices)
        
        # Always do visualization
        for sample_idx in sample_indices:
            print(f"Visualizing sample {sample_idx}...")
            analyzer.visualize_sample(sample_idx, n_slices=args.n_slices)
        
        # Generate report if analysis was run
        if not args.visualize_only:
            print("Generating report...")
            analyzer.generate_report()
    else:
        # Process single sample
        if not args.visualize_only:
            print(f"Analyzing sample {args.sample_idx}...")
            analyzer.analyze_sample(args.sample_idx)
        
        print(f"Visualizing sample {args.sample_idx}...")
        analyzer.visualize_sample(args.sample_idx, n_slices=args.n_slices)
        
        # Generate report if analysis was run
        if not args.visualize_only:
            print("Generating report...")
            analyzer.generate_report()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 