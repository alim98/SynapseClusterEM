import os
import glob
import numpy as np
import torch
from typing import Tuple
import imageio.v3 as iio
from scipy.ndimage import label
from torchvision import transforms
import matplotlib.pyplot as plt

# Import the config
from synapse.utils.config import config

class Synapse3DProcessor:
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            # Explicitly convert to grayscale with one output channel
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.mean = mean
        self.std = std
        self.normalize_volume = True  # New flag to control volume-wide normalization

    def __call__(self, frames, return_tensors=None):
        processed_frames = []
        for frame in frames:
            # Check if input is RGB (3 channels) or has unexpected shape
            if len(frame.shape) > 2 and frame.shape[2] > 1:
                if frame.shape[2] > 3:  # More than 3 channels
                    frame = frame[:, :, :3]  # Take first 3 channels
                # Will be converted to grayscale by the transform
            
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
            
        pixel_values = torch.stack(processed_frames)
        
        # Ensure we have a single channel
        if pixel_values.shape[1] != 1:
            # This should not happen due to transforms.Grayscale, but just in case
            pixel_values = pixel_values.mean(dim=1, keepdim=True)
        
        # Apply volume-wide normalization to ensure consistent grayscale values across slices
        if self.normalize_volume:
            # Method 1: Min-max normalization across the entire volume
            min_val = pixel_values.min()
            max_val = pixel_values.max()
            if max_val > min_val:  # Avoid division by zero
                pixel_values = (pixel_values - min_val) / (max_val - min_val)
            
            # Method 2: Alternative - Z-score normalization using mean and std
            # pixel_values = (pixel_values - pixel_values.mean()) / (pixel_values.std() + 1e-6)
            # pixel_values = torch.clamp((pixel_values * 0.5) + 0.5, 0, 1)  # Rescale to [0,1]
            
            # Method 3: Histogram matching across slices (would require more complex implementation)
            # This would ensure all slices have similar intensity distributions
            
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

    def save_segmented_slice(self, cube, output_path, slice_idx=None, consistent_gray=True):
        """
        Save a slice from a segmented cube with controlled normalization.
        
        Args:
            cube (numpy.ndarray): The cube with shape (y, x, c, z) from create_segmented_cube
            output_path (str): Path to save the image
            slice_idx (int, optional): Index of slice to save. If None, center slice is used.
            consistent_gray (bool): Whether to enforce consistent gray normalization
        """
        # Get the slice index (center if not specified)
        if slice_idx is None:
            slice_idx = cube.shape[3] // 2
        
        # Extract the slice - the cube is in (y, x, c, z) format
        slice_data = cube[:, :, :, slice_idx]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure with controlled normalization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
        if consistent_gray:
            ax.imshow(slice_data, vmin=0, vmax=1)
        else:
            ax.imshow(slice_data)
        
        ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        return output_path

class SynapseDataLoader:
    def __init__(self, raw_base_dir, seg_base_dir, add_mask_base_dir, gray_color=None):
        self.raw_base_dir = raw_base_dir
        self.seg_base_dir = seg_base_dir
        self.add_mask_base_dir = add_mask_base_dir
        # Use provided gray_color or get from config
        self.gray_color = gray_color if gray_color is not None else config.gray_color

    def load_volumes(self, bbox_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_dir = os.path.join(self.raw_base_dir, bbox_name)
        seg_dir = os.path.join(self.seg_base_dir, bbox_name)
        
        if bbox_name.startswith("bbox"):
            bbox_num = bbox_name.replace("bbox", "")
            add_mask_dir = os.path.join(self.add_mask_base_dir, f"bbox_{bbox_num}")
        else:
            add_mask_dir = os.path.join(self.add_mask_base_dir, bbox_name)
        
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
        
        if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
            return None, None, None
        
        try:
            # Load raw volume and convert to grayscale if needed
            raw_slices = []
            multi_channel_detected = False
            for f in raw_tif_files:
                img = iio.imread(f)
                # Check if the image has multiple channels (RGB)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # Convert RGB to grayscale using luminosity method
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                raw_slices.append(img)
            raw_vol = np.stack(raw_slices, axis=0)
            
            # Load segmentation volume and ensure it's single channel
            seg_slices = []
            for f in seg_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For segmentation, take first channel (labels should be consistent)
                    img = img[..., 0]
                seg_slices.append(img.astype(np.uint32))
            seg_vol = np.stack(seg_slices, axis=0)
            
            # Load additional mask volume and ensure it's single channel
            add_mask_slices = []
            for f in add_mask_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For masks, take first channel
                    img = img[..., 0]
                add_mask_slices.append(img.astype(np.uint32))
            add_mask_vol = np.stack(add_mask_slices, axis=0)
            
            if multi_channel_detected:
                print(f"WARNING: Multi-channel images detected in {bbox_name} and converted to single-channel")
            
            return raw_vol, seg_vol, add_mask_vol
        except Exception as e:
            print(f"Error loading volumes for {bbox_name}: {e}")
            return None, None, None

    @staticmethod
    def verify_single_channel(volume, name=""):
        """
        Verify if a volume is single-channel.
        
        Args:
            volume (numpy.ndarray): The volume to check
            name (str): Name for logging
            
        Returns:
            bool: True if single-channel, False otherwise
        """
        if len(volume.shape) == 3:  # Z, Y, X - single channel
            return True
            
        if len(volume.shape) == 4 and volume.shape[3] > 1:  # Z, Y, X, C with C > 1
            print(f"WARNING: Multi-channel volume detected in {name}: {volume.shape}")
            return False
            
        return True

    @staticmethod
    def get_closest_component_mask(full_mask, z_start, z_end, y_start, y_end, x_start, x_end, target_coord):
        sub_mask = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        labeled_sub_mask, num_features = label(sub_mask)
        
        if num_features == 0:
            return np.zeros_like(full_mask, dtype=bool)
        else:
            cx, cy, cz = target_coord
            min_distance = float('inf')
            closest_label = None

            for label_num in range(1, num_features + 1):
                vesicle_coords = np.column_stack(np.where(labeled_sub_mask == label_num))

                distances = np.sqrt(
                    (vesicle_coords[:, 0] + z_start - cz) ** 2 +
                    (vesicle_coords[:, 1] + y_start - cy) ** 2 +
                    (vesicle_coords[:, 2] + x_start - cx) ** 2
                )

                min_dist_for_vesicle = np.min(distances)
                if min_dist_for_vesicle < min_distance:
                    min_distance = min_dist_for_vesicle
                    closest_label = label_num

            if closest_label is not None:
                filtered_sub_mask = (labeled_sub_mask == closest_label)
                combined_mask = np.zeros_like(full_mask, dtype=bool)
                combined_mask[z_start:z_end, y_start:y_end, x_start:x_end] = filtered_sub_mask
                return combined_mask
            else:
                return np.zeros_like(full_mask, dtype=bool)

    def create_segmented_cube(
        self,
        raw_vol: np.ndarray,
        seg_vol: np.ndarray,
        add_mask_vol: np.ndarray,
        central_coord: Tuple[int, int, int],
        side1_coord: Tuple[int, int, int],
        side2_coord: Tuple[int, int, int],
        segmentation_type: int,
        subvolume_size: int = 80,
        alpha: float = 0.3,
        bbox_name: str = "",
        normalize_across_volume: bool = True,  # Add parameter to control normalization
        smart_crop: bool = False,  # New parameter to enable intelligent cropping
        presynapse_weight: float = 0.5,  # New parameter to control the shift toward presynapse (0.0-1.0)
    ) -> np.ndarray:
        bbox_num = bbox_name.replace("bbox", "").strip()
        
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

        # Original coordinates 
        cx, cy, cz = central_coord

        # Define a large temporary region to find presynapse components
        # This region should be larger than the final bounding box to allow for shifting
        temp_half_size = subvolume_size  # Double the size for initial analysis
        temp_x_start = max(cx - temp_half_size, 0)
        temp_x_end = min(cx + temp_half_size, raw_vol.shape[2])
        temp_y_start = max(cy - temp_half_size, 0)
        temp_y_end = min(cy + temp_half_size, raw_vol.shape[1])
        temp_z_start = max(cz - temp_half_size, 0)
        temp_z_end = min(cz + temp_half_size, raw_vol.shape[0])

        # Find vesicles in the expanded region
        vesicle_full_mask = (add_mask_vol == vesicle_label)
        temp_vesicle_mask = self.get_closest_component_mask(
            vesicle_full_mask,
            temp_z_start, temp_z_end,
            temp_y_start, temp_y_end,
            temp_x_start, temp_x_end,
            (cx, cy, cz)
        )

        def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
            x1, y1, z1 = s1_coord
            x2, y2, z2 = s2_coord
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            return mask_1, mask_2

        mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)

        # Determine which side is the presynapse by checking overlap with vesicle mask
        overlap_side1 = np.sum(np.logical_and(mask_1_full, temp_vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, temp_vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
        
        # Get presynapse mask
        presynapse_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        
        # When using smart cropping, adjust the center point toward the presynapse center of mass
        if smart_crop:
            # Get cleft mask for centering calculations
            cleft_mask_full = np.logical_or(
                (add_mask_vol == cleft_label),
                (add_mask_vol == cleft_label2)
            )
            
            # Extract the region of interest in the temp bounding box
            temp_presynapse_mask = presynapse_mask_full[temp_z_start:temp_z_end, 
                                                        temp_y_start:temp_y_end, 
                                                        temp_x_start:temp_x_end]
            
            # Calculate center of mass of presynapse within the temp region
            if np.any(temp_presynapse_mask):
                presynapse_coords = np.array(np.where(temp_presynapse_mask)).T
                presynapse_com = np.mean(presynapse_coords, axis=0)
                
                # Convert center of mass to global coordinates
                presynapse_com_global = np.array([
                    presynapse_com[0] + temp_z_start,
                    presynapse_com[1] + temp_y_start,
                    presynapse_com[2] + temp_x_start
                ])
                
                # Weighted average between cleft center (original central_coord) and presynapse center of mass
                # presynapse_weight controls how much we shift toward the presynapse (0.0-1.0)
                adjusted_center = np.array([cz, cy, cx]) * (1 - presynapse_weight) + presynapse_com_global * presynapse_weight
                
                # Update the cropping center coordinates
                cz, cy, cx = adjusted_center.astype(int)
                
                # Print for debugging
                print(f"Smart cropping: Original center: ({central_coord}), Adjusted center: ({cx, cy, cz})")
                print(f"Shifted by: {np.array([cx, cy, cz]) - np.array([central_coord[0], central_coord[1], central_coord[2]])}")
        
        # Calculate the final bounding box with possibly adjusted center
        half_size = subvolume_size // 2
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, raw_vol.shape[2])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, raw_vol.shape[1])
        z_start = max(cz - half_size, 0)
        z_end = min(cz + half_size, raw_vol.shape[0])

        # Get vesicle mask for the final bounding box region
        vesicle_mask = self.get_closest_component_mask(
            vesicle_full_mask,
            z_start, z_end,
            y_start, y_end,
            x_start, x_end,
            (cx, cy, cz)
        )

        def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
            x1, y1, z1 = s1_coord
            x2, y2, z2 = s2_coord
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            return mask_1, mask_2

        mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)

        overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2

        if segmentation_type == 0:
            combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
        elif segmentation_type == 1:
            combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        elif segmentation_type == 2:
            combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
        elif segmentation_type == 3:
            combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
        elif segmentation_type == 4:
            vesicle_closest = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest2 = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest,cleft_closest2))
        elif segmentation_type == 5:
            vesicle_closest = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_extra = np.logical_or(vesicle_closest, cleft_closest)
            combined_mask_full = np.logical_or(mask_1_full, np.logical_or(mask_2_full, combined_mask_extra))
        elif segmentation_type == 6:
            combined_mask_full = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
        elif segmentation_type == 7:
            cleft_closest = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest2 = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_full =  np.logical_or(cleft_closest,cleft_closest2)
        elif segmentation_type == 8:
            combined_mask_full = self.get_closest_component_mask(
                (add_mask_vol == mito_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
        elif segmentation_type == 10:
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
            combined_mask_full = np.logical_or(cleft_closest,pre_mask_full)
        elif segmentation_type == 9:
            vesicle_closest = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            combined_mask_full = np.logical_or(cleft_closest,vesicle_closest)
        else:
            raise ValueError(f"Unsupported segmentation type: {segmentation_type}")

        sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
        sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]

        pad_z = subvolume_size - sub_raw.shape[0]
        pad_y = subvolume_size - sub_raw.shape[1]
        pad_x = subvolume_size - sub_raw.shape[2]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            sub_combined_mask = np.pad(sub_combined_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)

        sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
        sub_combined_mask = sub_combined_mask[:subvolume_size, :subvolume_size, :subvolume_size]

        sub_raw = sub_raw.astype(np.float32)
        
        # Apply normalization across the entire volume or per slice
        if normalize_across_volume:
            # Global normalization across the entire volume
            min_val = np.min(sub_raw)
            max_val = np.max(sub_raw)
            range_val = max_val - min_val if max_val > min_val else 1.0
            normalized = (sub_raw - min_val) / range_val
            
            # Print for debugging
            print(f"Global normalization: min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
        else:
            # Original per-slice normalization
            mins = np.min(sub_raw, axis=(1, 2), keepdims=True)
            maxs = np.max(sub_raw, axis=(1, 2), keepdims=True)
            ranges = np.where(maxs > mins, maxs - mins, 1.0)
            normalized = (sub_raw - mins) / ranges
            
            # Print for debugging
            print(f"Per-slice normalization: shape of mins={mins.shape}, maxs={maxs.shape}")

        # Convert to RGB here ONLY for visualization purposes
        # The data processing pipeline uses grayscale (1-channel) format
        raw_rgb = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
        mask_factor = sub_combined_mask[..., np.newaxis]

        if alpha < 1:
            blended_part = alpha * self.gray_color + (1 - alpha) * raw_rgb
        else:
            blended_part = self.gray_color * (1 - mask_factor) + raw_rgb * mask_factor

        overlaid_image = raw_rgb * mask_factor + (1 - mask_factor) * blended_part

        overlaid_cube = np.transpose(overlaid_image, (1, 2, 3, 0))

        return overlaid_cube

    def save_segmented_slice(self, cube, output_path, slice_idx=None, consistent_gray=True):
        """
        Save a slice from a segmented cube with controlled normalization.
        
        Args:
            cube (numpy.ndarray): The cube with shape (y, x, c, z) from create_segmented_cube
            output_path (str): Path to save the image
            slice_idx (int, optional): Index of slice to save. If None, center slice is used.
            consistent_gray (bool): Whether to enforce consistent gray normalization
        """
        # Get the slice index (center if not specified)
        if slice_idx is None:
            slice_idx = cube.shape[3] // 2
        
        # Extract the slice - the cube is in (y, x, c, z) format
        slice_data = cube[:, :, :, slice_idx]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure with controlled normalization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
        if consistent_gray:
            ax.imshow(slice_data, vmin=0, vmax=1)
        else:
            ax.imshow(slice_data)
        
        ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        return output_path

def normalize_cube_globally(cube):
    """
    Apply global normalization to a cube to ensure consistent grayscale values across slices.
    
    Args:
        cube (numpy.ndarray): The cube to normalize, expected to be in format (y, x, c, z)
        
    Returns:
        numpy.ndarray: Normalized cube with consistent grayscale values
    """
    # Make a copy to avoid modifying the original
    cube_copy = cube.copy()
    
    # Calculate global min and max across all dimensions
    min_val = np.min(cube_copy)
    max_val = np.max(cube_copy)
    
    # Avoid division by zero
    if max_val > min_val:
        # Apply global normalization
        cube_copy = (cube_copy - min_val) / (max_val - min_val)
        
    return cube_copy 