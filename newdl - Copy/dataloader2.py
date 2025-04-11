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
        self.normalize_volume = False  # New flag to control volume-wide normalization

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
        normalize_presynapse_size: bool = False,  # New parameter to enable presynapse size normalization
        target_percentage: float = None,  # Target percentage of presynapse pixels (None = use mean)
        size_tolerance: float = 0.1,  # Tolerance range as a percentage of the target (±10% by default)
        vesicle_fill_threshold: float = 100.0,  # Required percentage of vesicle fill for type 12 (default 95%)
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
        
        # Get the presynapse mask for size normalization
        presynapse_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full

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
        elif segmentation_type == 11: # segtype 10 without mito
            # Get cleft mask
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
            all_mito_mask = (add_mask_vol == mito_label)
            
            # Dilate the mitochondria mask by 2 voxels to create a safety margin
            from scipy import ndimage
            dilated_mito_mask = ndimage.binary_dilation(all_mito_mask, iterations=2)
            
            combined_temp = np.logical_or(cleft_closest, pre_mask_full)
            # Exclude dilated mitochondria mask from the combined mask
            combined_mask_full = np.logical_and(combined_temp, np.logical_not(dilated_mito_mask))
        elif segmentation_type == 12: # vescile cloud with 25x25x25 bounding box
            # Get the vesicle cloud mask
            vesicle_cloud = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            
            # Initial check - make sure we have any vesicle pixels at all
            vesicle_pixel_count = np.sum(vesicle_cloud)
            if vesicle_pixel_count == 0:
                print(f"Discarding sample: No vesicle pixels found at all (0 pixels)")
                return None
                
            print(f"Initial vesicle cloud size: {vesicle_pixel_count} pixels")
            
            # Check if we have enough vesicle pixels to potentially fill a 25×25×25 box
            box_size = 25
            required_pixels = box_size * box_size * box_size  # Exactly 15,625 pixels needed
            if vesicle_pixel_count < required_pixels:
                print(f"Discarding sample: Vesicle cloud too small ({vesicle_pixel_count} pixels < {required_pixels} required)")
                return None
            
            # Get the vesicle cloud coordinates
            vesicle_coords = np.where(vesicle_cloud)
            if len(vesicle_coords[0]) == 0:
                print(f"Error: No vesicle coordinates found despite having {vesicle_pixel_count} pixels")
                return None
                
            # Find the bounding box of the vesicle cloud
            z_min, z_max = np.min(vesicle_coords[0]), np.max(vesicle_coords[0])
            y_min, y_max = np.min(vesicle_coords[1]), np.max(vesicle_coords[1])
            x_min, x_max = np.min(vesicle_coords[2]), np.max(vesicle_coords[2])
            
            # Check if the vesicle cloud is large enough in each dimension to fit a box_size box
            z_size = z_max - z_min + 1
            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            
            print(f"Vesicle cloud dimensions: {z_size}×{y_size}×{x_size}")
            
            if z_size < box_size or y_size < box_size or x_size < box_size:
                print(f"Discarding sample: Vesicle cloud dimensions too small to fit {box_size}×{box_size}×{box_size} box")
                return None
            
            # Using sliding window approach to find a position where the box is 100% filled
            found_fully_filled_box = False
            best_box_coords = None
            best_box_fill_percentage = 0.0
            
            # Potential starting positions for the sliding window, we'll try different positions
            # Limit the search space to save computation time
            max_positions_per_dim = 10  # Limit to 10 positions per dimension to avoid excessive computation
            
            # Calculate step sizes for each dimension
            z_step = max(1, (z_size - box_size) // max_positions_per_dim)
            y_step = max(1, (y_size - box_size) // max_positions_per_dim)
            x_step = max(1, (x_size - box_size) // max_positions_per_dim)
            
            print(f"Searching for a fully filled box with steps: z={z_step}, y={y_step}, x={x_step}")
            
            # Create a dense 3D boolean array representing the vesicle cloud
            # This is more efficient for the sliding window
            voxel_grid = np.zeros((z_max-z_min+1, y_max-y_min+1, x_max-x_min+1), dtype=bool)
            for i in range(len(vesicle_coords[0])):
                z, y, x = vesicle_coords[0][i], vesicle_coords[1][i], vesicle_coords[2][i]
                voxel_grid[z-z_min, y-y_min, x-x_min] = True
            
            # Try different starting positions for the box with sliding window approach
            positions_checked = 0
            for z_start in range(z_min, z_max - box_size + 2, z_step):
                for y_start in range(y_min, y_max - box_size + 2, y_step):
                    for x_start in range(x_min, x_max - box_size + 2, x_step):
                        positions_checked += 1
                        
                        # Check if the box at this position is completely filled
                        z_end = z_start + box_size
                        y_end = y_start + box_size
                        x_end = x_start + box_size
                        
                        # Skip if box extends beyond volume bounds
                        if z_end > vesicle_cloud.shape[0] or y_end > vesicle_cloud.shape[1] or x_end > vesicle_cloud.shape[2]:
                            continue
                            
                        # Extract the sub-grid for this box from the voxel grid
                        sub_grid = voxel_grid[z_start-z_min:z_end-z_min, 
                                             y_start-y_min:y_end-y_min, 
                                             x_start-x_min:x_end-x_min]
                        
                        # Count filled voxels
                        filled_count = np.sum(sub_grid)
                        fill_percentage = filled_count / required_pixels * 100.0
                        
                        # Update best box if this one has a higher fill percentage
                        if fill_percentage > best_box_fill_percentage:
                            best_box_fill_percentage = fill_percentage
                            best_box_coords = (z_start, z_end, y_start, y_end, x_start, x_end)
                            
                            # If 100% filled, we can stop searching
                            if fill_percentage >= 100.0:
                                found_fully_filled_box = True
                                break
                                
                    if found_fully_filled_box:
                        break
                if found_fully_filled_box:
                    break
            
            print(f"Checked {positions_checked} positions. Best fill percentage: {best_box_fill_percentage:.2f}%")
            
            # Use threshold from the vesicle_fill_threshold parameter (default 100.0%)
            required_fill_percentage = vesicle_fill_threshold
            
            if best_box_fill_percentage < required_fill_percentage:
                print(f"Discarding sample: Could not find a box with {required_fill_percentage:.1f}% fill (best was {best_box_fill_percentage:.2f}%)")
                return None
                
            # Create the combined mask using the best box found
            z_start, z_end, y_start, y_end, x_start, x_end = best_box_coords
            
            # Create a box mask for the final bounding box
            combined_mask_full = np.zeros_like(vesicle_cloud, dtype=bool)
            box_with_vesicles = np.logical_and(
                np.ones((z_end-z_start, y_end-y_start, x_end-x_start), dtype=bool),
                vesicle_cloud[z_start:z_end, y_start:y_end, x_start:x_end]
            )
            combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end] = box_with_vesicles
            
            print(f"Successfully created {box_size}×{box_size}×{box_size} bounding box with {best_box_fill_percentage:.2f}% fill")
        elif segmentation_type == 13: # vescile cloud with 25x25x25 bounding box + pre + cleft
            # Get the vesicle cloud mask
            vesicle_cloud = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            
            # Get the presynaptic side mask
            pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
            
            # Get cleft mask
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
            )
            
            # Get mitochondria mask and dilate it to create safety margin
            all_mito_mask = (add_mask_vol == mito_label)
            
            # Dilate the mitochondria mask by 2 voxels to create a safety margin
            from scipy import ndimage
            dilated_mito_mask = ndimage.binary_dilation(all_mito_mask, iterations=2)
            
            # Create a 25×25×25 bounding box around the center of the vesicle cloud
            # without shifting toward the center of synapse
            vesicle_coords = np.where(vesicle_cloud)
            if len(vesicle_coords[0]) > 0:
                # Find the center of the vesicle cloud
                v_cz = int(np.mean(vesicle_coords[0]))
                v_cy = int(np.mean(vesicle_coords[1]))
                v_cx = int(np.mean(vesicle_coords[2]))
                
                # Create a new empty mask
                box_mask = np.zeros_like(vesicle_cloud, dtype=bool)
                
                # Define bounds for the 25×25×25 box
                box_size = 12  # Half of 25 (12 on each side plus the center point)
                z_min = max(0, v_cz - box_size)
                z_max = min(box_mask.shape[0], v_cz + box_size + 1)
                y_min = max(0, v_cy - box_size)
                y_max = min(box_mask.shape[1], v_cy + box_size + 1)
                x_min = max(0, v_cx - box_size)
                x_max = min(box_mask.shape[2], v_cx + box_size + 1)
                
                # Set the box region to True
                box_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True
                
                # Combine vesicle, pre, and cleft masks
                combined_masks = np.logical_or(vesicle_cloud, np.logical_or(pre_mask_full, cleft_closest))
                
                # Exclude dilated mitochondria from the combined masks
                combined_masks = np.logical_and(combined_masks, np.logical_not(dilated_mito_mask))
                
                # Intersection of combined masks and the box - include everything within the box
                combined_mask_full = np.logical_and(combined_masks, box_mask)
                
                print(f"Type 13 box with center at ({v_cx}, {v_cy}, {v_cz}) includes vesicle, pre, and cleft masks, excluding mito")
            else:
                # If no vesicle cloud is found, try to place box at the center of pre+cleft
                pre_cleft_mask = np.logical_or(pre_mask_full, cleft_closest)
                
                # Exclude dilated mitochondria from the pre+cleft mask
                pre_cleft_mask = np.logical_and(pre_cleft_mask, np.logical_not(dilated_mito_mask))
                
                pre_cleft_coords = np.where(pre_cleft_mask)
                
                if len(pre_cleft_coords[0]) > 0:
                    # Find the center of the pre+cleft mask
                    pc_cz = int(np.mean(pre_cleft_coords[0]))
                    pc_cy = int(np.mean(pre_cleft_coords[1]))
                    pc_cx = int(np.mean(pre_cleft_coords[2]))
                    
                    # Create a 25×25×25 box around this center
                    box_mask = np.zeros_like(vesicle_cloud, dtype=bool)
                    
                    # Define bounds for the 25×25×25 box
                    box_size = 12  # Half of 25 (12 on each side plus the center point)
                    z_min = max(0, pc_cz - box_size)
                    z_max = min(box_mask.shape[0], pc_cz + box_size + 1)
                    y_min = max(0, pc_cy - box_size)
                    y_max = min(box_mask.shape[1], pc_cy + box_size + 1)
                    x_min = max(0, pc_cx - box_size)
                    x_max = min(box_mask.shape[2], pc_cx + box_size + 1)
                    
                    # Set the box region to True
                    box_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True
                    
                    # Intersection of combined masks and the box
                    combined_mask_full = np.logical_and(pre_cleft_mask, box_mask)
                    
                    print(f"Type 13 fallback: Using pre+cleft center at ({pc_cx}, {pc_cy}, {pc_cz}), excluding mito")
                else:
                    # If nothing else works, use central coordinate
                    box_mask = np.zeros_like(vesicle_cloud, dtype=bool)
                    
                    # Define bounds for the 25×25×25 box
                    box_size = 12
                    z_min = max(0, cz - box_size)
                    z_max = min(box_mask.shape[0], cz + box_size + 1)
                    y_min = max(0, cy - box_size)
                    y_max = min(box_mask.shape[1], cy + box_size + 1)
                    x_min = max(0, cx - box_size)
                    x_max = min(box_mask.shape[2], cx + box_size + 1)
                    
                    # Set the box region to True
                    box_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True
                    
                    # Use all available masks in this region, excluding mitochondria
                    all_masks = np.logical_or(vesicle_cloud, np.logical_or(pre_mask_full, cleft_closest))
                    all_masks = np.logical_and(all_masks, np.logical_not(dilated_mito_mask))
                    combined_mask_full = np.logical_and(all_masks, box_mask)
                    
                    print(f"Type 13 fallback: Using central coordinate at ({cx}, {cy}, {cz}), excluding mito")
  
        else:
            raise ValueError(f"Unsupported segmentation type: {segmentation_type}")

        sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
        sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Extract the presynapse mask for the final bounding box
        sub_presynapse_mask = presynapse_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Apply presynapse size normalization if enabled
        if normalize_presynapse_size and np.any(sub_presynapse_mask):
            print(f"Applying presynapse size normalization for {bbox_name}")
            # Calculate the current percentage of presynapse pixels in the cube
            total_pixels = sub_presynapse_mask.size
            presynapse_pixels = np.sum(sub_presynapse_mask)
            current_percentage = presynapse_pixels / total_pixels
            
            # Use the provided target percentage or set to current (no change)
            if target_percentage is None:
                target_percentage = current_percentage
                print(f"Using current percentage as target: {target_percentage:.4f}")
            else:
                print(f"Using provided target percentage: {target_percentage:.4f}")
            
            # Calculate the lower and upper bounds of the acceptable range
            lower_bound = target_percentage * (1 - size_tolerance)
            upper_bound = target_percentage * (1 + size_tolerance)
            
            print(f"Presynapse size: Current {current_percentage:.4f}, Target {target_percentage:.4f}, Range [{lower_bound:.4f}-{upper_bound:.4f}]")
            print(f"Presynapse pixels: {presynapse_pixels} out of {total_pixels} total pixels")
            
            # Only adjust if outside the acceptable range
            if current_percentage < lower_bound or current_percentage > upper_bound:
                # Get the coordinates of the presynapse voxels
                presynapse_coords = np.array(np.where(sub_presynapse_mask)).T
                
                if len(presynapse_coords) > 0:
                    print(f"Found {len(presynapse_coords)} presynapse coordinates")
                    # Calculate the centroid of the presynapse
                    centroid = np.mean(presynapse_coords, axis=0)
                    print(f"Centroid: {centroid}")
                    
                    # Calculate distance of each voxel from the centroid
                    distances = np.sqrt(np.sum((presynapse_coords - centroid)**2, axis=1))
                    
                    # Sort distances to enable manipulation from outside in or inside out
                    sorted_indices = np.argsort(distances)
                    sorted_coords = presynapse_coords[sorted_indices]
                    
                    if current_percentage > upper_bound:
                        # Presynapse is too large - shrink it by removing outer voxels
                        # Calculate how many voxels to remove
                        target_voxels = int(total_pixels * target_percentage)
                        voxels_to_remove = presynapse_pixels - target_voxels
                        
                        print(f"Presynapse too large: removing {voxels_to_remove} voxels (target: {target_voxels})")
                        
                        # Remove voxels from outside in (largest distances first)
                        for i in range(int(voxels_to_remove)):
                            if i < len(sorted_coords):
                                # Start from the end (furthest from centroid)
                                coord = sorted_coords[-(i+1)]
                                sub_presynapse_mask[coord[0], coord[1], coord[2]] = False
                        
                        print(f"Presynapse shrunk: removed {voxels_to_remove} voxels")
                        
                    elif current_percentage < lower_bound:
                        # Presynapse is too small - grow it by adding neighboring voxels
                        # Calculate how many voxels to add
                        target_voxels = int(total_pixels * target_percentage)
                        voxels_to_add = target_voxels - presynapse_pixels
                        
                        print(f"Presynapse too small: adding {voxels_to_add} voxels (target: {target_voxels})")
                        
                        # Create a distance map from the presynapse
                        # First, create a copy of the mask and expand it
                        expanded_mask = sub_presynapse_mask.copy()
                        added_voxels = 0
                        
                        # Grow by dilation (adding neighboring voxels layer by layer)
                        from scipy import ndimage
                        
                        # Continue dilating until we've added enough voxels or can't add more
                        iterations = 0
                        max_iterations = 10  # Prevent infinite loops
                        
                        while added_voxels < voxels_to_add and iterations < max_iterations:
                            # Dilate the mask by 1 voxel
                            dilated = ndimage.binary_dilation(expanded_mask)
                            
                            # Find new voxels (in dilated but not in original)
                            new_voxels = np.logical_and(dilated, ~expanded_mask)
                            
                            # Count new voxels
                            num_new = np.sum(new_voxels)
                            print(f"Iteration {iterations}: Found {num_new} new voxels")
                            
                            if num_new == 0:
                                # No more voxels can be added (reached the boundary)
                                print("No more voxels can be added - reached boundary")
                                break
                                
                            # If adding all new voxels would exceed the target, select a subset
                            if added_voxels + num_new > voxels_to_add:
                                # Get indices of new voxels
                                new_coords = np.array(np.where(new_voxels)).T
                                
                                # Calculate distances from centroid
                                new_distances = np.sqrt(np.sum((new_coords - centroid)**2, axis=1))
                                
                                # Sort by distance (closest first)
                                new_sorted_indices = np.argsort(new_distances)
                                
                                # Select only the needed number of voxels (closest to centroid)
                                voxels_needed = int(voxels_to_add - added_voxels)
                                selected_coords = new_coords[new_sorted_indices[:voxels_needed]]
                                
                                print(f"Selecting {voxels_needed} closest voxels from {len(new_coords)} candidates")
                                
                                # Clear the new voxels mask and set only selected voxels
                                new_voxels = np.zeros_like(new_voxels)
                                for coord in selected_coords:
                                    new_voxels[coord[0], coord[1], coord[2]] = True
                                
                                num_new = voxels_needed
                            
                            # Update the presynapse mask with the new voxels
                            sub_presynapse_mask = np.logical_or(sub_presynapse_mask, new_voxels)
                            
                            # Update the expanded mask for next iteration
                            expanded_mask = dilated
                            
                            # Update the count of added voxels
                            added_voxels += num_new
                            iterations += 1
                        
                        print(f"Presynapse expanded: added {added_voxels} voxels in {iterations} iterations")
                else:
                    print("Error: No presynapse coordinates found despite mask having True values")
                
                # Recalculate the percentage after adjustment
                # new_percentage = np.sum(sub_presynapse_mask) / total_pixels
                # print(f"Adjusted presynapse size: {new_percentage:.4f} (target was {target_percentage:.4f})")
                
                # Update the combined mask with the normalized presynapse
                if segmentation_type == 1:  # Segmentation type that only uses presynapse
                    # For segmentation types that directly use the presynapse mask
                    print(f"Segmentation type {segmentation_type} directly uses presynapse - replacing combined mask")
                    sub_combined_mask = sub_presynapse_mask
                elif segmentation_type == 10:  # Special handling for type 10 which combines cleft and presynapse
                    # For segmentation type 10, we need to preserve the cleft part and update only the presynapse part
                    print(f"Segmentation type 10 combines cleft and presynapse - preserving cleft while updating presynapse")
                    
                    # Get the original combined mask from the full volume
                    orig_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Extract just the cleft part by subtracting the original presynapse mask
                    # First, get the presynapse mask before normalization
                    orig_presynapse = presynapse_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Then, extract just the cleft part (everything in combined mask that's not in the presynapse)
                    cleft_only = np.logical_and(orig_combined_mask, ~orig_presynapse)
                    
                    # Now combine the cleft part with the normalized presynapse
                    sub_combined_mask = np.logical_or(cleft_only, sub_presynapse_mask)
                    
                    print(f"Updated combined mask for segmentation type 10 - preserved cleft and updated presynapse")
                else:
                    # For segmentation types that include presynapse as part of a larger mask
                    # Remove the original presynapse from the combined mask and add the normalized one
                    print(f"Segmentation type {segmentation_type} includes presynapse as part - updating combined mask")
                    sub_combined_mask = np.logical_or(
                        np.logical_and(sub_combined_mask, ~sub_presynapse_mask),  # Other parts without presynapse
                        sub_presynapse_mask  # Add normalized presynapse
                    )

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
            # Use the raw values directly WITHOUT any normalization to ensure absolute consistent gray values
            # Just clip to a reasonable range if values are too extreme
            # This is crucial for consistent gray levels across different samples
            
            # Define fixed min/max clip values for all samples - THESE MUST REMAIN CONSTANT
            fixed_min = 0.0
            fixed_max = 255.0
            
            # If raw values are already in 0-1 range, scale to 0-255 for processing
            if np.max(sub_raw) <= 1.0:
                sub_raw = sub_raw * 255.0
            
            # Clip values to fixed range
            clipped_raw = np.clip(sub_raw, fixed_min, fixed_max)
            
            # Scale to 0-1 for visualization
            normalized = clipped_raw / 255.0
            
            print(f"Using ABSOLUTE scaling with FIXED values: min={fixed_min}, max={fixed_max}")
            print(f"Raw range before fixed clipping: {np.min(sub_raw):.4f}-{np.max(sub_raw):.4f}")

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

    def get_vesicle_cloud_info(
        self,
        raw_vol: np.ndarray,
        seg_vol: np.ndarray,
        add_mask_vol: np.ndarray,
        central_coord: Tuple[int, int, int],
        side1_coord: Tuple[int, int, int],
        side2_coord: Tuple[int, int, int],
        subvolume_size: int = 80,
        bbox_name: str = "",
    ) -> dict:
        """
        Extract vesicle cloud information without rendering the sample.
        This is used for analysis purposes to determine why samples may be discarded.
        
        Returns:
            dict: Dictionary with vesicle cloud information:
                - vesicle_pixel_count: Number of vesicle cloud pixels
                - total_subvol_pixels: Total number of pixels in the subvolume
                - vesicle_portion: Ratio of vesicle cloud pixels to total pixels
                - has_enough_vesicle_pixels: Whether there are enough vesicle pixels for a 25×25×25 box
                - best_box_fill_percentage: The best fill percentage we could find for a 25×25×25 box
                - discard_reason: Specific reason why this sample would be discarded (if any)
        """
        try:
            # Determine vesicle label based on bounding box
            bbox_num = bbox_name.replace("bbox", "").strip()
            
            if bbox_num in {'2', '5',}:
                vesicle_label = 3
            elif bbox_num == '7':
                vesicle_label = 2
            elif bbox_num == '4':
                vesicle_label = 2
            elif bbox_num == '3':
                vesicle_label = 7
            else:
                vesicle_label = 6
            
            # Extract vesicle cloud mask from add_mask_vol
            vesicle_cloud_mask = add_mask_vol == vesicle_label
            
            # Extract the coordinates for the subvolume
            cx, cy, cz = central_coord
            
            # Calculate the bounds for the subvolume
            half_size = subvolume_size // 2
            
            z_start = max(0, cz - half_size)
            z_end = min(raw_vol.shape[0], cz + half_size)
            y_start = max(0, cy - half_size)
            y_end = min(raw_vol.shape[1], cy + half_size)
            x_start = max(0, cx - half_size)
            x_end = min(raw_vol.shape[2], cx + half_size)
            
            # Get the closest vesicle cloud component
            vesicle_cloud = self.get_closest_component_mask(
                vesicle_cloud_mask,
                z_start, z_end,
                y_start, y_end,
                x_start, x_end,
                (cx, cy, cz)
            )
            
            # Count vesicle cloud pixels and total pixels
            vesicle_pixel_count = np.sum(vesicle_cloud)
            total_subvol_pixels = vesicle_cloud.size
            
            # Calculate the portion of vesicle cloud pixels
            vesicle_portion = vesicle_pixel_count / total_subvol_pixels if total_subvol_pixels > 0 else 0
            
            # Set up the initial result
            result = {
                'vesicle_pixel_count': int(vesicle_pixel_count),
                'total_subvol_pixels': int(total_subvol_pixels),
                'vesicle_portion': float(vesicle_portion),
                'has_enough_vesicle_pixels': False,
                'best_box_fill_percentage': 0.0,
                'discard_reason': None
            }
            
            # Check if there are any vesicle pixels at all
            if vesicle_pixel_count == 0:
                result['discard_reason'] = "No vesicle cloud pixels found"
                return result
            
            # Define the box size for vesicle cloud type 12
            box_size = 25
            required_pixels = box_size * box_size * box_size  # 15,625 pixels for a 25×25×25 box
            
            # Check if we have enough vesicle pixels to potentially fill a box
            if vesicle_pixel_count < required_pixels:
                result['discard_reason'] = f"Vesicle cloud too small ({vesicle_pixel_count} pixels < {required_pixels} required)"
                return result
            
            # We have enough pixels, now check if we can fit a box
            result['has_enough_vesicle_pixels'] = True
            
            # Get the vesicle cloud coordinates
            vesicle_coords = np.where(vesicle_cloud)
            
            # Find the bounding box of the vesicle cloud
            z_min, z_max = np.min(vesicle_coords[0]), np.max(vesicle_coords[0])
            y_min, y_max = np.min(vesicle_coords[1]), np.max(vesicle_coords[1])
            x_min, x_max = np.min(vesicle_coords[2]), np.max(vesicle_coords[2])
            
            # Check the dimensions of the vesicle cloud
            z_size = z_max - z_min + 1
            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            
            # Check if the vesicle cloud is large enough in each dimension to fit a box_size box
            if z_size < box_size or y_size < box_size or x_size < box_size:
                result['discard_reason'] = f"Vesicle cloud dimensions too small to fit {box_size}×{box_size}×{box_size} box (dimensions: {z_size}×{y_size}×{x_size})"
                return result
            
            # Create a dense 3D boolean array representing the vesicle cloud
            voxel_grid = np.zeros((z_max-z_min+1, y_max-y_min+1, x_max-x_min+1), dtype=bool)
            for i in range(len(vesicle_coords[0])):
                z, y, x = vesicle_coords[0][i], vesicle_coords[1][i], vesicle_coords[2][i]
                voxel_grid[z-z_min, y-y_min, x-x_min] = True
            
            # Try different starting positions for the box to find the best fill percentage
            best_box_fill_percentage = 0.0
            
            # Limit search space for analysis purposes
            max_positions_per_dim = 5
            
            # Calculate step sizes for each dimension
            z_step = max(1, (z_size - box_size) // max_positions_per_dim)
            y_step = max(1, (y_size - box_size) // max_positions_per_dim)
            x_step = max(1, (x_size - box_size) // max_positions_per_dim)
            
            # Search for best box position
            for z_start in range(z_min, z_max - box_size + 2, z_step):
                for y_start in range(y_min, y_max - box_size + 2, y_step):
                    for x_start in range(x_min, x_max - box_size + 2, x_step):
                        # Extract the subgrid for this box position
                        sub_grid = voxel_grid[z_start-z_min:z_start-z_min+box_size, 
                                             y_start-y_min:y_start-y_min+box_size, 
                                             x_start-x_min:x_start-x_min+box_size]
                        
                        # Count filled voxels
                        filled_count = np.sum(sub_grid)
                        fill_percentage = filled_count / required_pixels * 100.0
                        
                        # Update best fill percentage
                        if fill_percentage > best_box_fill_percentage:
                            best_box_fill_percentage = fill_percentage
            
            result['best_box_fill_percentage'] = best_box_fill_percentage
            
            # Check if best fill percentage meets the threshold for rendering
            # We'll use a default threshold of 95.0% but this should match the value in create_segmented_cube
            if best_box_fill_percentage < 99.0:
                result['discard_reason'] = f"Could not find a box with sufficient fill percentage (best: {best_box_fill_percentage:.2f}% < 99.0% required)"
            
            return result
            
        except Exception as e:
            print(f"Error extracting vesicle cloud information: {e}")
            import traceback
            traceback.print_exc()
            return {
                'vesicle_pixel_count': 0,
                'total_subvol_pixels': 0,
                'vesicle_portion': 0.0,
                'has_enough_vesicle_pixels': False,
                'best_box_fill_percentage': 0.0,
                'discard_reason': f"Error: {str(e)}"
            }

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