# Model Input Visualization Tool

This tool visualizes the exact inputs that the neural network model receives during training and inference. It's designed to help validate model inputs and understand what features the model is learning from.

## Purpose

The main purposes of this tool are:

1. **Model Input Validation**: Verify that the data preprocessing pipeline is working correctly and the model is receiving the expected inputs.
2. **Feature Understanding**: Visualize what features are being presented to the model to better understand its learning process.
3. **Debugging**: Identify potential issues in the data processing pipeline by examining the actual model inputs.
4. **Documentation**: Generate visualizations for reports, presentations, and documentation.

## Features

- Visualizes center slices of 3D volumes as static images
- Creates animated GIFs to show the 3D nature of the data
- Supports different segmentation types (raw, presynapse, postsynapse, etc.)
- Adjustable alpha blending for visualization
- Consistent gray values for non-segmented regions
- Generates a summary file with sample information

## Usage

### Basic Usage

Run the tool with default settings:

**On Windows:**
```
cd scripts\model_input_visualization
.\run_visualization.cmd
```

**On Linux/Mac:**
```bash
cd scripts/model_input_visualization
./run_visualization.sh
```

This will:
- Process 100 random samples from all bounding boxes
- Use segmentation type 1 (presynapse) with alpha 1.0
- Apply a consistent gray value of 0.6 to non-segmented regions
- Save results to `outputs/model_input_visualization`

### Advanced Usage

You can customize the visualization with various parameters:

**On Windows:**
```
.\run_visualization.cmd --segmentation_type 1 --alpha 1.0 --bbox_names bbox1 bbox2 --num_samples 50 --fixed_gray_value 0.5
```

**On Linux/Mac:**
```bash
./run_visualization.sh --segmentation_type 1 --alpha 1.0 --bbox_names bbox1 bbox2 --num_samples 50 --fixed_gray_value 0.5
```

**Direct Python Usage:**
```bash
python visualize_inputs.py --segmentation_type 1 --alpha 1.0 --bbox_names bbox1 bbox2 --num_samples 50 --fixed_gray_value 0.5
```

### Parameters

- `--raw_base_dir`: Base directory for raw data
- `--seg_base_dir`: Base directory for segmentation data
- `--add_mask_base_dir`: Base directory for additional mask data
- `--excel_dir`: Directory containing Excel files
- `--output_dir`: Directory to save output files
- `--bbox_names`: Bounding box names to include (space-separated)
- `--num_samples`: Number of samples to visualize
- `--sample_indices`: Specific sample indices to visualize (optional)
- `--segmentation_type`: Type of segmentation to use:
  - 0: Raw data
  - 1: Presynapse
  - 2: Postsynapse
  - 3: Both sides
  - 4: Vesicles + cleft
- `--alpha`: Alpha value for blending (0.0-1.0)
- `--fixed_gray_value`: Fixed gray value for non-segmented regions (0.0-1.0)
- `--use_global_norm`: Use global normalization for visualization

## Output

The tool generates:

1. **Center frame images**: PNG files showing the center slice of each sample
2. **Animated GIFs**: GIF files showing all frames of each sample
3. **Summary file**: Text file with detailed information about all samples

All outputs are saved to the specified output directory (default: `outputs/model_input_visualization`).

## Example

To visualize 20 samples from bbox1 and bbox2 with segmentation type 1, alpha 0.5, and a fixed gray value of 0.7:

**On Windows:**
```
.\run_visualization.cmd --bbox_names bbox1 bbox2 --num_samples 20 --segmentation_type 1 --alpha 0.5 --fixed_gray_value 0.7
```

**On Linux/Mac:**
```bash
./run_visualization.sh --bbox_names bbox1 bbox2 --num_samples 20 --segmentation_type 1 --alpha 0.5 --fixed_gray_value 0.7
```

## Consistent Gray Values

By default, the tool applies a consistent fixed gray value (0.6) to non-segmented regions. This ensures that the gray color is the same across all samples, making it easier to compare them. You can adjust this value using the `--fixed_gray_value` parameter. 