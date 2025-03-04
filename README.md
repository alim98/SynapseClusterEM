# SynapseClusterEM

A Python package for analyzing and clustering 3D synapse structures from electron microscopy (EM) data using deep learning techniques.

## Features

- Load and process 3D synapse volumes from EM data
- Extract features using a pre-trained VGG3D model
- Perform clustering analysis on extracted features
- Generate 2D and 3D visualizations of synapse clusters
- Support for multiple segmentation types and bounding boxes

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

## Project Structure

```
SynapseClusterEM/
├── synapse_analysis/
│   ├── models/
│   │   └── vgg3d.py         # VGG3D model implementation
│   ├── data/
│   │   ├── dataset.py       # Dataset class
│   │   └── data_loader.py   # Data loading utilities
│   ├── utils/
│   │   └── processing.py    # Image processing utilities
│   └── analysis/
│       ├── clustering.py    # Clustering analysis
│       └── feature_extraction.py  # Feature extraction
├── scripts/
│   ├── run_analysis.py      # Main analysis script
│   ├── run_notebook_analysis.py  # Analysis script for notebooks
│   ├── run_notebook_analysis.sh  # Shell script for notebook analysis
│   ├── visualize_sample_as_gif.py  # Visualization script for samples
│   ├── visualize_sample_as_gif.sh  # Shell script for visualization
│   └── workflow.sh          # Complete analysis pipeline script
├── outputs/
│   └── gif_visualization/   # Output directory for GIF visualizations
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

Data preprocessing involves loading and preparing the raw, segmentation, and additional mask volumes for analysis.

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
    --output_dir outputs/analysis
```

This step:
- Loads a pre-trained VGG3D model
- Extracts features from the samples
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
1. Run feature extraction, clustering, and visualization
2. Visualize representative samples from each cluster

Before running, make sure to edit the script to set the correct paths for your environment:
- `RAW_BASE_DIR`: Path to raw data
- `SEG_BASE_DIR`: Path to segmentation data
- `ADD_MASK_BASE_DIR`: Path to additional mask data
- `EXCEL_DIR`: Path to Excel files
- `CHECKPOINT_PATH`: Path to VGG3D model checkpoint

Alternatively, you can run each step manually:

```bash
# 1. Extract features, perform clustering, and generate visualizations
python scripts/run_analysis.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --add_mask_base_dir /path/to/mask/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path path/to/vgg3d_checkpoint.pth \
    --bbox_names bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7 \
    --segmentation_types 1 2 3 \
    --alphas 1.0 \
    --output_dir outputs/analysis

# 2. Visualize specific samples of interest
./scripts/visualize_sample_as_gif.sh \
    --bbox_name bbox1 \
    --sample_index 0 \
    --segmentation_type 1
``` 