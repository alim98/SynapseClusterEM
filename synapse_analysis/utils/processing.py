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
    # Extract region of interest
    region = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # If no components are present, return an empty mask
    if not np.any(region):
        return np.zeros_like(full_mask, dtype=bool)
    
    # Label connected components
    labeled_region, num_components = label(region)
    if num_components == 0:
        return np.zeros_like(full_mask, dtype=bool)
    
    # Get target coordinate relative to ROI
    tx, ty, tz = target_coord
    tx_roi = tx - x_start
    ty_roi = ty - y_start
    tz_roi = tz - z_start
    
    # Ensure target coordinates are within bounds
    tx_roi = max(0, min(tx_roi, region.shape[2] - 1))
    ty_roi = max(0, min(ty_roi, region.shape[1] - 1))
    tz_roi = max(0, min(tz_roi, region.shape[0] - 1))
    
    # Compute distances for each labeled component
    component_dists = {}
    for i in range(1, num_components + 1):
        if i not in component_dists:  # This component not encountered yet
            component_mask = (labeled_region == i)
            where_result = np.where(component_mask)
            z_indices, y_indices, x_indices = where_result
            
            # Compute distances to each voxel in the component
            distances = np.sqrt(
                (x_indices - tx_roi) ** 2 +
                (y_indices - ty_roi) ** 2 +
                (z_indices - tz_roi) ** 2
            )
            
            # Record the minimum distance
            component_dists[i] = np.min(distances)
    
    # Find component with minimum distance
    if not component_dists:  # No components
        return np.zeros_like(full_mask, dtype=bool)
    
    closest_component = min(component_dists, key=component_dists.get)
    
    # Create the final mask
    result_mask = np.zeros_like(full_mask, dtype=bool)
    closest_mask_roi = (labeled_region == closest_component)
    result_mask[z_start:z_end, y_start:y_end, x_start:x_end] = closest_mask_roi
    
    return result_mask

def get_bbox_labels(bbox_name: str) -> Dict[str, int]:
    """
    Get the labels for different structures based on the bounding box name.
    
    Args:
        bbox_name: Name of the bounding box
        
    Returns:
        Dictionary with labels for different structures
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
    else:  # For bbox1, bbox6, etc.
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
                         bbox_name: str = "",
                         global_mean: float = None,
                         global_std: float = None) -> np.ndarray:
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
        global_mean: Global mean for normalization (deprecated, not used)
        global_std: Global std for normalization (deprecated, not used)
        
    Returns:
        Segmented cube as a numpy array
    """
    labels = get_bbox_labels(bbox_name)
    
    # --- Always calculate subvolume bounds FIRST ---
    half_size = subvolume_size // 2
    cx, cy, cz = central_coord
    x_start = max(cx - half_size, 0)
    x_end = min(cx + half_size, raw_vol.shape[2])
    y_start = max(cy - half_size, 0)
    y_end = min(cy + half_size, raw_vol.shape[1])
    z_start = max(cz - half_size, 0)
    z_end = min(cz + half_size, raw_vol.shape[0])

    # --- Vesicle filtering (critical for presynapse determination) ---
    vesicle_full_mask = (add_mask_vol == labels['vesicle_label'])
    vesicle_mask = get_closest_component_mask(
        vesicle_full_mask,
        z_start, z_end,
        y_start, y_end,
        x_start, x_end,
        (cx, cy, cz)
    )

    # --- Side masks ---
    def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        seg_id_1 = segmentation_volume[z1, y1, x1]
        seg_id_2 = segmentation_volume[z2, y2, x2]
        mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        return mask_1, mask_2

    mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)

    # --- Determine pre-synapse side using filtered vesicles ---
    overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
    overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
    presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
    # print(f"overlap_side1={overlap_side1}_overlap_side2={overlap_side2}_side1_coord:{side1_coord}_side1_coord:{side2_coord}")
    # --- Segmentation type handling ---
    if segmentation_type == 0: # Raw data
        combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
    elif segmentation_type == 1:  # Presynapse
        combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
    elif segmentation_type == 2:  # Postsynapse
        combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
    elif segmentation_type == 3:  # Both sides
        combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
    elif segmentation_type == 4:  # Vesicles + Cleft (closest only)
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == labels['vesicle_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest = get_closest_component_mask(
            ((add_mask_vol == labels['cleft_label'])), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest2 = get_closest_component_mask(
            ((add_mask_vol == labels['cleft_label2'])), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest,cleft_closest2))
    elif segmentation_type == 5:  # (closest vesicles/cleft + sides)
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == labels['vesicle_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_extra = np.logical_or(vesicle_closest, cleft_closest)
        combined_mask_full = np.logical_or(mask_1_full, np.logical_or(mask_2_full, combined_mask_extra))
    elif segmentation_type == 6:  # Vesicle cloud (closest)
        combined_mask_full = get_closest_component_mask(
            (add_mask_vol == labels['vesicle_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
    elif segmentation_type == 7:  # Cleft (closest)
        cleft_closest = get_closest_component_mask(
            ((add_mask_vol == labels['cleft_label'])), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest2 = get_closest_component_mask(
            ((add_mask_vol == labels['cleft_label2'])), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_full =  np.logical_or(cleft_closest,cleft_closest2)
    elif segmentation_type == 8:  # Mitochondria (closest)
        combined_mask_full = get_closest_component_mask(
            (add_mask_vol == labels['mito_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
    elif segmentation_type == 10:  #  +Cleft +pre
        # vesicle_closest = get_closest_component_mask(
        #     (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        # )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full

        combined_mask_full = np.logical_or(cleft_closest,pre_mask_full)

    elif segmentation_type == 9:  # cleft+vesicle
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == labels['vesicle_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label']), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        # pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full

        combined_mask_full = np.logical_or(cleft_closest,vesicle_closest)

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
    
    # Convert to float32
    sub_raw = sub_raw.astype(np.float32)
    
    # Log raw data statistics
    print(f"[DEBUG] Raw data stats for {bbox_name}:")
    print(f"[DEBUG] - Shape: {sub_raw.shape}")
    print(f"[DEBUG] - Min value: {np.min(sub_raw)}")
    print(f"[DEBUG] - Max value: {np.max(sub_raw)}")
    print(f"[DEBUG] - Mean value: {np.mean(sub_raw)}")
    print(f"[DEBUG] - Std dev: {np.std(sub_raw)}")
    
    # Log mask statistics
    mask_coverage = np.sum(sub_combined_mask) / sub_combined_mask.size * 100
    print(f"[DEBUG] Mask stats for {bbox_name}:")
    print(f"[DEBUG] - Mask coverage: {mask_coverage:.2f}%")
    
    # First normalize the raw data to 0-255 range for consistent visualization
    min_val = np.min(sub_raw)
    max_val = np.max(sub_raw)
    
    # Avoid division by zero
    if max_val > min_val:
        normalized_raw = 255.0 * (sub_raw - min_val) / (max_val - min_val)
        print(f"[DEBUG] Normalized data from range [{min_val}, {max_val}] to [0, 255]")
    else:
        normalized_raw = np.zeros_like(sub_raw)
        print(f"[DEBUG] Warning: Uniform intensity image (min=max={min_val}), set to zeros")
    
    # Log normalized data statistics
    print(f"[DEBUG] Normalized data stats:")
    print(f"[DEBUG] - Min value: {np.min(normalized_raw)}")
    print(f"[DEBUG] - Max value: {np.max(normalized_raw)}")
    print(f"[DEBUG] - Mean value: {np.mean(normalized_raw)}")
    
    # Now apply mask with fixed gray value (128 is middle of 0-255 range)
    gray_value = 128.0
    print(f"[DEBUG] Using gray value: {gray_value}")
    
    # Apply mask overlay
    result = np.where(sub_combined_mask, normalized_raw, gray_value)
    
    # Log result statistics
    print(f"[DEBUG] Result stats after masking:")
    print(f"[DEBUG] - Min value: {np.min(result)}")
    print(f"[DEBUG] - Max value: {np.max(result)}")
    print(f"[DEBUG] - Mean value: {np.mean(result)}")
    print(f"[DEBUG] - Unique values in masked regions: {np.unique(result[~sub_combined_mask])}")
    
    # Add a dimension to match expected output format [H, W, 1, D]
    # This maintains compatibility with existing code but uses only 1 channel
    result = result[..., np.newaxis]
    
    # Transpose dimensions to match expected output format [H, W, C, D]
    overlaid_cube = np.transpose(result, (0, 1, 3, 2))
    
    print(f"[DEBUG] Final output shape: {overlaid_cube.shape}")
    print(f"[DEBUG] ----------------------------------------")
    
    return overlaid_cube 