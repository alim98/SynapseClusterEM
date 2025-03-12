# Synapse Visualization Enhancements Documentation

## Overview

We have implemented several significant enhancements to the synapse visualization system to improve feature extraction and analysis. These enhancements include intelligent cropping functionality, improved visualization techniques, and better comparative analysis tools.

## 1. Intelligent Cropping Functionality

### Purpose
The original system cropped synapses based solely on the center of the cleft. The new intelligent cropping considers both the cleft and presynapse position, which is crucial for effective feature extraction.

### Implementation Details
- **Smart Crop Parameter**: Added a `smart_crop` boolean parameter to enable/disable intelligent cropping
- **Presynapse Weight Parameter**: Added a `presynapse_weight` parameter (range 0.0-1.0) to control how much the bounding box is shifted toward the presynapse
- **Center Point Calculation**: Implemented a weighted average between the original cleft center and the presynapse center of mass
- **Shift Vector Reporting**: Added logging and visualization of the calculated shift vector

### Benefits
- Better capture of presynaptic features
- Ensures the cleft remains within the bounding box
- Configurable via weight parameter to adjust for different analysis needs

## 1.1 Presynapse Size Normalization

### Purpose
Presynapses in the dataset can vary significantly in size. Some occupy a large portion of the cube while others are small, making comparisons difficult. The size normalization feature adjusts presynapses to have more consistent relative sizes.

### Implementation Details
- **Normalization Parameters**:
  - `normalize_presynapse_size`: Boolean to enable/disable size normalization
  - `target_percentage`: Target percentage of cube volume for presynapse (None = use mean)
  - `size_tolerance`: Tolerance range (±%) for acceptable sizes

- **Size Calculation**:
  - Computes the percentage of voxels occupied by the presynapse in each cube
  - Determines if a presynapse is larger or smaller than the target size

- **Resizing Algorithm**:
  - **For Large Presynapses**: Shrinks by removing outer voxels (furthest from centroid)
  - **For Small Presynapses**: Grows by adding adjacent voxels through controlled dilation
  - Preserves structural integrity by adding/removing voxels based on distance from centroid

- **Integration with Segmentation**:
  - Updates the segmentation mask with the normalized presynapse
  - Maintains structural relationships with other elements (cleft, postsynapse)

### Benefits
- More consistent presynapse proportions across samples
- Facilitates better comparison between different synapses
- Reduces bias from outlier presynapse sizes in analysis
- Retains structural integrity of the presynaptic region

## 2. Visualization Improvements

### Purpose
Enhanced the visual representation to make comparisons clearer and more informative.

### Implementation Details
- **Improved Layout**:
  - Added margins between images
  - Created header space for better title display
  - Added separator lines between panels
  - Used consistent background colors

- **Text Enhancements**:
  - Centered titles over respective images
  - Added text outlines for better visibility
  - Truncated long titles with ellipsis when necessary
  - Added frame counters for animation progress

- **Consistent Grayscale**:
  - Implemented global normalization across all frames
  - Eliminated per-slice normalization that caused inconsistent gray values
  - Applied consistent scaling based on volume-wide min/max values

### Benefits
- More professional and readable visualizations
- Consistent appearance across all frames
- Better visual comparison between standard and intelligent cropping

## 3. Multi-Weight Comparison Tool

### Purpose
Created a tool to directly compare the effects of different presynapse weights.

### Implementation Details
- **Multi-Panel Display**:
  - Created a new function `create_multi_weight_comparison`
  - Shows standard cropping alongside multiple weight options in a single GIF
  - Dynamically arranges panels based on available weights

- **Automatic Processing**:
  - Tracks generated GIFs for each weight
  - Creates combined comparison automatically

- **Visual Metadata**:
  - Weight value labeled on each panel
  - Sample information displayed in header
  - Frame counter included for reference

### Benefits
- Direct visual comparison of different weight settings
- Easier parameter tuning
- Better understanding of how weight affects the cropping behavior

## 3.1 Size Normalization Comparison Tool

### Purpose
Created a tool to directly compare the effects of presynapse size normalization on both standard and intelligent cropping methods.

### Implementation Details
- **Four-Panel Display**:
  - Created a new function `create_size_normalization_comparison`
  - Shows four versions in a 2x2 grid:
    - Standard cropping without normalization
    - Standard cropping with size normalization
    - Intelligent cropping without normalization
    - Intelligent cropping with size normalization

- **Visual Organization**:
  - Clear panel headers indicating normalization status
  - Main title showing sample information
  - Consistent layout with borders and spacing
  - Frame counter for animation tracking

- **Automatic Processing**:
  - Generates all variants with a single configuration
  - Creates the comparison visualization automatically

### Benefits
- Direct visual assessment of size normalization effects
- Easy comparison of how normalization affects both cropping methods
- Better understanding of size variation impact on feature visibility

## 4. Performance and Quality Improvements

### Purpose
Enhanced the overall quality and user experience.

### Implementation Details
- **Frame Rate Adjustment**:
  - Reduced FPS from 10 to 8 for better visualization
  - Makes it easier to observe details in the animations

- **Error Handling**:
  - Added comprehensive error logging
  - Created fallbacks for font loading
  - Included detailed exception reporting

- **Metadata Generation**:
  - Created companion info files with detailed parameters
  - Logged shift vectors and other key metrics
  - Included range information for post-analysis

### Benefits
- More stable and reliable visualization process
- Better documentation of parameters for reproducibility
- Enhanced user experience when viewing the GIFs

## 5. Technical Implementation

### Code Structure
- `create_segmented_cube`: Modified to support intelligent cropping and size normalization
- `visualize_comparison`: Enhanced for better visualization and metadata
- `create_combined_frames`: Improved layout and appearance
- `create_multi_weight_comparison`: Function for multi-weight comparison
- `create_size_normalization_comparison`: Function for size normalization comparison
- `main`: Updated to orchestrate the generation of all comparison types

### Workflow
1. Load volume data and synapse information
2. Generate cubes with different cropping and normalization settings
3. Process frames with consistent normalization
4. Create comparison visualizations (standard vs. intelligent)
5. Generate multi-weight comparisons
6. Generate size normalization comparisons
7. Save detailed metadata for analysis

## Usage Examples

### Running Basic Comparison
```python
python -m newdl.sample_fig_compare_crop
```

### Modifying Presynapse Weights
To experiment with different weight values, modify the following line in the main function:
```python
presynapse_weights = [0.3, 0.5, 0.7]  # Different weights to compare
```

### Controlling Size Normalization
To adjust size normalization parameters, update these values in the `visualize_comparison` call:
```python
normalize_presynapse_size=True,
target_percentage=0.15,  # Target 15% of cube volume
size_tolerance=0.1  # Accept ±10% variation
```

### Output Files
The script generates the following types of output files:
- `bbox_name_var1_standard.gif` - Standard cropping
- `bbox_name_var1_standard_normalized.gif` - Standard cropping with size normalization
- `bbox_name_var1_intelligent_w{weight}.gif` - Intelligent cropping with specific weight
- `bbox_name_var1_intelligent_w{weight}_normalized.gif` - Intelligent cropping with size normalization
- `bbox_name_var1_comparison_w{weight}.gif` - Side-by-side comparison
- `bbox_name_var1_comparison_w{weight}_normalized.gif` - Side-by-side comparison with size normalization
- `bbox_name_var1_multi_weight_comparison.gif` - Multi-weight comparison
- `bbox_name_var1_size_normalization_comparison.gif` - Four-panel size normalization comparison
- `bbox_name_var1_info.txt` - Metadata about the comparison

## Conclusion

These enhancements significantly improve the functionality and usability of the synapse visualization system. The intelligent cropping provides better feature extraction by considering presynapse positioning, while size normalization ensures more consistent presynapse proportions across samples. The improved visualization tools make it easier to analyze and compare different cropping strategies and normalization effects. The multi-weight and size normalization comparisons offer powerful ways to tune parameters and understand the impact of different settings. 

## 6. Bug Fixes and Improvements

### 6.1 Segmentation Type 10 Size Normalization Fix

#### Issue
When using segmentation type 10 (which combines both cleft and presynapse structures) with size normalization enabled, only a circular region was displayed instead of the expected combined visualization of both structures.

#### Root Cause
The bug occurred because the size normalization process was completely replacing the combined mask (which should include both cleft and presynapse) with just the normalized presynapse mask. This caused the visualization to show only the normalized presynapse portion without the cleft.

#### Implementation Fix
Updated the handling of segmentation type 10 in the dataloader to:
1. Preserve the cleft part from the original mask
2. Update only the presynapse part with the normalized version 
3. Recombine both structures correctly after normalization

```python
# Updated code for segmentation type 10 handling
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
```

#### Benefits
- Correctly visualizes both cleft and presynapse structures when size normalization is enabled
- Maintains the structural relationship between cleft and presynapse components
- Ensures consistent visualization across all segmentation types 