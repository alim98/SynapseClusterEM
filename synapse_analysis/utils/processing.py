import numpy as np
from scipy.ndimage import label
from typing import Tuple, Dict

def get_closest_component_mask(full_mask: np.ndarray,
                             z_start: int, z_end: int,
                             y_start: int, y_end: int,
                             x_start: int, x_end: int,
                             target_coord: Tuple[int, int, int]) -> np.ndarray:
    """
    Get the mask for the component closest to the target coordinate.
    
    Args:
        full_mask: Binary mask of the full volume
        z_start, z_end, y_start, y_end, x_start, x_end: Coordinates defining the region of interest
        target_coord: Target coordinate (x, y, z) to find the closest component to
        
    Returns:
        Binary mask of the closest component
    """
    sub_mask = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
    labeled_sub_mask, num_features = label(sub_mask)
    
    if num_features == 0:
        return np.zeros_like(full_mask, dtype=bool)
        
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

def get_bbox_labels(bbox_name: str) -> Dict[str, int]:
    """
    Get the label mappings for a specific bounding box.
    
    Args:
        bbox_name: Name of the bounding box
        
    Returns:
        Dictionary containing label mappings for the bbox
    """
    bbox_num = bbox_name.replace("bbox", "").strip()
    
    if bbox_num in {'2', '5'}:
        return {
            'mito_label': 1,
            'vesicle_label': 3,
            'cleft_label': 2,
            'cleft_label2': 4
        }
    elif bbox_num == '7':
        return {
            'mito_label': 1,
            'vesicle_label': 2,
            'cleft_label': 4,
            'cleft_label2': 3
        }
    elif bbox_num == '4':
        return {
            'mito_label': 3,
            'vesicle_label': 2,
            'cleft_label': 1,
            'cleft_label2': 4
        }
    elif bbox_num == '3':
        return {
            'mito_label': 6,
            'vesicle_label': 7,
            'cleft_label': 9,
            'cleft_label2': 8
        }
    else:  # For bbox1, 6, etc.
        return {
            'mito_label': 5,
            'vesicle_label': 6,
            'cleft_label': 7,
            'cleft_label2': 7
        }

def create_segmented_cube(raw_vol: np.ndarray,
                         seg_vol: np.ndarray,
                         add_mask_vol: np.ndarray,
                         central_coord: Tuple[int, int, int],
                         side1_coord: Tuple[int, int, int],
                         side2_coord: Tuple[int, int, int],
                         segmentation_type: int,
                         subvolume_size: int = 80,
                         alpha: float = 0.3,
                         bbox_name: str = "") -> np.ndarray:
    """
    Create a segmented cube from the input volumes.
    
    Args:
        raw_vol: Raw image volume
        seg_vol: Segmentation volume
        add_mask_vol: Additional mask volume
        central_coord: Central coordinate (x, y, z)
        side1_coord: Side 1 coordinate (x, y, z)
        side2_coord: Side 2 coordinate (x, y, z)
        segmentation_type: Type of segmentation to apply
        subvolume_size: Size of the output cube
        alpha: Alpha blending factor
        bbox_name: Name of the bounding box
        
    Returns:
        Segmented cube as a numpy array
    """
    labels = get_bbox_labels(bbox_name)
    
    # Calculate bounds
    half_size = subvolume_size // 2
    cx, cy, cz = central_coord
    x_start = max(cx - half_size, 0)
    x_end = min(cx + half_size, raw_vol.shape[2])
    y_start = max(cy - half_size, 0)
    y_end = min(cy + half_size, raw_vol.shape[1])
    z_start = max(cz - half_size, 0)
    z_end = min(cz + half_size, raw_vol.shape[0])
    
    # Get vesicle mask
    vesicle_full_mask = (add_mask_vol == labels['vesicle_label'])
    vesicle_mask = get_closest_component_mask(
        vesicle_full_mask, z_start, z_end, y_start, y_end, x_start, x_end, central_coord
    )
    
    # Create segment masks
    def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        seg_id_1 = segmentation_volume[z1, y1, x1]
        seg_id_2 = segmentation_volume[z2, y2, x2]
        mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        return mask_1, mask_2
    
    mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)
    
    # Determine pre-synapse side
    overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
    overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
    presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
    
    # Create combined mask based on segmentation type
    if segmentation_type == 0:  # Raw data
        combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
    elif segmentation_type == 1:  # Presynapse
        combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
    elif segmentation_type == 2:  # Postsynapse
        combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
    elif segmentation_type == 3:  # Both sides
        combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
    elif segmentation_type in [4, 9]:  # Vesicles + Cleft
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == labels['vesicle_label']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        cleft_closest2 = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label2']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest, cleft_closest2))
    else:
        raise ValueError(f"Unsupported segmentation type: {segmentation_type}")
    
    # Extract and process subvolume
    sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Padding if needed
    pad_z = subvolume_size - sub_raw.shape[0]
    pad_y = subvolume_size - sub_raw.shape[1]
    pad_x = subvolume_size - sub_raw.shape[2]
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        sub_combined_mask = np.pad(sub_combined_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)
    
    sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_combined_mask = sub_combined_mask[:subvolume_size, :subvolume_size, :subvolume_size]
    
    # Convert to float32 and normalize
    normalized = sub_raw.astype(np.float32)
    gray_color = 0.6
    
    # Create RGB version and apply masking
    raw_rgb = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
    mask_factor = sub_combined_mask[..., np.newaxis]
    
    if alpha < 1:
        blended_part = alpha * gray_color + (1 - alpha) * raw_rgb
    else:
        blended_part = gray_color * (1 - mask_factor) + raw_rgb * mask_factor
    
    overlaid_image = raw_rgb * mask_factor + (1 - mask_factor) * blended_part
    
    # Transpose dimensions
    overlaid_cube = np.transpose(overlaid_image, (1, 2, 3, 0))
    
    return overlaid_cube 