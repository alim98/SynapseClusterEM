# SynapseClusterEM

A Python package for analyzing and clustering 3D synapse structures from electron microscopy (EM) data using deep learning techniques.

## Features

- Load and process 3D synapse volumes from EM data
- Extract features using a pre-trained VGG3D model
- Perform clustering analysis on extracted features
- Generate 2D and 3D visualizations of synapse clusters
- Support for multiple segmentation types and bounding boxes
- **Global data normalization** across the entire dataset for improved model performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alim98/SynapseClusterEM.git
cd SynapseClusterEM
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

Run the analysis script with your data:

```bash
python scripts/run_analysis.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path path/to/checkpoint \
    --bbox_names bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7 \
    --segmentation_types 9 10 \
    --alphas 1.0 \
    --output_dir outputs
```

### Arguments

- `--raw_base_dir`: Directory containing raw image data
- `--seg_base_dir`: Directory containing segmentation data
- `--add_mask_base_dir`: Directory containing additional mask data
- `--excel_dir`: Directory containing Excel files with synapse information
- `--checkpoint_path`: Path to the VGG3D model checkpoint
- `--bbox_names`: List of bounding box names to process
- `--segmentation_types`: List of segmentation types to analyze
- `--alphas`: List of alpha values for blending
- `--output_dir`: Directory to save results
- `--batch_size`: Batch size for processing (default: 4)
- `--num_workers`: Number of worker processes (default: 2)
- `--use_global_norm`: Whether to use global normalization (default: False)
- `--num_samples_for_stats`: Number of samples to use for calculating global stats (default: 100, use 0 for all)

## Global Normalization

The package now includes support for global normalization of data. This method calculates the mean and standard deviation across the entire dataset, rather than using fixed values. Global normalization typically leads to better model performance, especially when working with datasets that have varying intensity distributions.

There are two methods for calculating global normalization statistics:

### Method 1: Direct Volume-Based Normalization

This method calculates statistics directly from the raw volumes, which is faster and more memory-efficient:

```bash
python scripts/global_norm_example.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --output_dir outputs/global_norm
```

### Method 2: Dataset-Based Normalization

This method calculates statistics from processed samples in the dataset:

```bash
python scripts/global_normalization_example.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --output_dir outputs \
    --num_samples_for_stats 100
```

Both methods will:
1. Calculate global mean and standard deviation
2. Save the statistics for future use
3. Create a data processor that applies global normalization

The global statistics are saved to JSON files for reuse in future runs.

### Using Global Normalization in Your Code

To use global normalization in your own code:

```python
from synapse_analysis.data.data_loader import Synapse3DProcessor, load_all_volumes

# Load your volume data
vol_data_dict = load_all_volumes(bbox_names, raw_base_dir, seg_base_dir, add_mask_base_dir)

# Method 1: Calculate global stats directly from volumes
processor = Synapse3DProcessor.create_with_global_norm_from_volumes(vol_data_dict)

# Method 2: Load pre-calculated global stats
import json
with open('path/to/global_stats.json', 'r') as f:
    global_stats = json.load(f)
    
processor = Synapse3DProcessor(apply_global_norm=True, global_stats=global_stats)
```

## Project Structure

```
SynapseClusterEM/
├── synapse_analysis/
│   ├── models/
│   │   └── vgg3d.py         # VGG3D model implementation
│   ├── data/
│   │   ├── dataset.py       # Dataset class
│   │   └── data_loader.py   # Data loading utilities with global normalization
│   ├── utils/
│   │   └── processing.py    # Image processing utilities
│   └── analysis/
│       ├── clustering.py    # Clustering analysis
│       └── feature_extraction.py  # Feature extraction
├── scripts/
│   ├── run_analysis.py      # Main analysis script
│   ├── run_notebook_analysis.py  # Analysis script for notebooks
│   ├── run_notebook_analysis.sh  # Shell script for notebook analysis
│   ├── global_norm_example.py  # Global normalization example
│   ├── global_norm_example.sh  # Shell script for global normalization
│   ├── visualize_sample_as_gif.py  # Visualization script for samples
│   ├── visualize_sample_as_gif.sh  # Shell script for visualization
│   └── workflow.sh          # Complete analysis pipeline script
├── outputs/
│   ├── gif_visualization/   # Output directory for GIF visualizations
│   └── global_norm/         # Output directory for global normalization statistics
├── requirements.txt         # Package dependencies
└── setup.py                # Package setup file
```

## Requirements

- Python 3.6+
- PyTorch
- scikit-learn
- UMAP
- Plotly
- Other dependencies listed in requirements.txt 

## Workflow

This project follows a clear workflow for analyzing 3D synapse structures:

### 1. Data Preprocessing

```bash
# Preprocess data with global normalization
python scripts/global_norm_example.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --output_dir outputs/global_norm \
    --segmentation_type 1
```

This step:
- Loads raw, segmentation, and additional mask volumes
- Calculates global normalization statistics across all volumes
- Saves the statistics to `outputs/global_norm/global_stats.json`
- Creates a dataset with globally normalized samples

### 2. Feature Extraction

```bash
# Extract features using a pre-trained VGG3D model
python scripts/run_analysis.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path path/to/vgg3d_checkpoint.pth \
    --bbox_names bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7 \
    --segmentation_types 1 2 3 \
    --alphas 1.0 \
    --output_dir outputs/analysis \
    --use_global_norm True \
    --global_stats_path outputs/global_norm/global_stats.json
```

This step:
- Uses the globally normalized dataset
- Loads a pre-trained VGG3D model
- Extracts features from the normalized samples
- Saves the extracted features to `outputs/analysis/features.pkl`

### 3. Dimensionality Reduction & Visualization

The analysis script automatically performs:
- UMAP projection of the extracted features
- 2D scatter plots of the UMAP projections
- 3D interactive visualizations using Plotly
- Saving visualizations to `outputs/analysis/visualizations/`

### 4. Clustering Analysis

The analysis script also performs:
- K-means clustering on the extracted features
- Hierarchical clustering for comparison
- Visualization of cluster assignments on UMAP projections
- Statistical analysis of clusters
- Saving clustering results to `outputs/analysis/clustering/`

### 5. Sample Visualization

```bash
# Visualize specific samples as GIFs
python scripts/visualize_sample_as_gif.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --bbox_name bbox1 \
    --sample_index 0 \
    --segmentation_type 1 \
    --alpha 1.0 \
    --gray_value 0.5 \
    --fps 10 \
    --output_dir outputs/gif_visualization
```

This step:
- Creates GIF visualizations of specific samples
- Applies segmentation masks with customizable parameters
- Saves GIFs to `outputs/gif_visualization/`

## Complete Analysis Pipeline

For a complete analysis pipeline, you can use the provided workflow script:

```bash
# Run the complete analysis pipeline
./scripts/workflow.sh
```

This script will:
1. Calculate global normalization statistics
2. Run feature extraction, clustering, and visualization
3. Visualize representative samples from each cluster

Before running, make sure to edit the script to set the correct paths for your environment:
- `RAW_BASE_DIR`: Path to raw data
- `SEG_BASE_DIR`: Path to segmentation data
- `ADD_MASK_BASE_DIR`: Path to additional mask data
- `EXCEL_DIR`: Path to Excel files
- `CHECKPOINT_PATH`: Path to VGG3D model checkpoint

Alternatively, you can run each step manually:

```bash
# 1. Preprocess data with global normalization
./scripts/global_norm_example.sh

# 2. Extract features, perform clustering, and generate visualizations
python scripts/run_analysis.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path path/to/vgg3d_checkpoint.pth \
    --bbox_names bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7 \
    --segmentation_types 1 2 3 \
    --alphas 1.0 \
    --output_dir outputs/analysis \
    --use_global_norm True \
    --global_stats_path outputs/global_norm/global_stats.json

# 3. Visualize specific samples of interest
./scripts/visualize_sample_as_gif.sh \
    --bbox_name bbox1 \
    --sample_index 0 \
    --segmentation_type 1
``` 