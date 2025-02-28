# Synapse Analysis

A Python package for analyzing 3D synapse structures using deep learning and clustering techniques.

## Features

- Load and process 3D synapse volumes
- Extract features using a pre-trained VGG3D model
- Perform clustering analysis on extracted features
- Generate 2D and 3D visualizations of synapse clusters
- Support for multiple segmentation types and bounding boxes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/synapse_analysis.git
cd synapse_analysis
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
synapse_analysis/
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
│   └── run_analysis.py      # Main analysis script
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