# Normalization Guide for Synapse Visualization

## Problem Description

When visualizing synapse data with masked regions, we encountered two main issues:

### 1. Different intensity ranges in each slice
- Even though we normalized the entire 3D volume to 0-255, each 2D slice had a different intensity range
- For example, one slice had values from 8.9 to 184.8, while another had values from 8.3 to 243.1
- This meant that the gray value (128.0) appeared different relative to the data in each slice

### 2. Matplotlib's automatic scaling
- Even though we set vmin=0, vmax=255, the visual contrast was still affected by the actual data range in each slice
- This made the gray value appear inconsistent across different images

## Solution: Two-Step Normalization Process

We implemented a two-step normalization process to ensure consistent visualization:

### First Normalization in 3D
1. We normalize the entire 3D volume to 0-255 range
2. We apply the mask with a fixed gray value of 128.0

### Second Normalization in 2D
1. For each 2D slice, we identify the masked regions (where value is exactly 128.0)
2. We temporarily replace the gray value with a special value (-1000)
3. We normalize only the non-masked regions to 0-255 range
4. We restore the gray value to 128.0

This ensures that the gray value is always exactly in the middle of the 0-255 range relative to the actual data in each slice, providing consistent visualization across all images.

## Implementation Details

The implementation of this approach can be found in:
- `main.py`: Contains the 2D slice normalization for visualization
- `synapse_analysis/utils/processing.py`: Contains the 3D volume normalization

## Usage

When visualizing slices from the 3D volume:

```python
# Example code for proper visualization with consistent gray values
# 1. Extract a 2D slice from the 3D volume
slice_img = central_slice[:, :, central_slice.shape[2] // 2]

# 2. Identify the masked regions (gray value = 128.0)
gray_mask = np.isclose(slice_img, 128.0)

# 3. Normalize the non-masked regions
normalized_slice = slice_img.copy()
normalized_slice[gray_mask] = -1000  # Temporary special value
non_gray_mask = ~gray_mask

if np.any(non_gray_mask):
    min_val = np.min(normalized_slice[non_gray_mask])
    max_val = np.max(normalized_slice[non_gray_mask])
    
    if max_val > min_val:
        normalized_slice[non_gray_mask] = (normalized_slice[non_gray_mask] - min_val) / (max_val - min_val) * 255

# 4. Restore the gray value
normalized_slice[gray_mask] = 128.0

# 5. Display with fixed range
plt.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
```

## Benefits

- Consistent visual appearance of masked regions across all images
- Improved contrast for the actual data in each slice
- Better visual interpretation of the results 