# SynapseClusterEM

A deep learning framework for analyzing and clustering 3D synapse structures from electron microscopy (EM) data, developed at the Max Planck Institute for Brain Research in Frankfurt, Germany.

## Overview

SynapseClusterEM is a powerful tool designed to analyze 3D synapse morphology using advanced neural networks and unsupervised learning techniques. The project aims to identify structural patterns and classify synapses based on their 3D architecture, providing insights into brain connectivity and function.

![GUI Screenshot](assets/scr.png)

## Features

- Intuitive graphical user interface (GUI) for easy configuration and operation
- 3D synapse image data processing from electron microscopy datasets
- Feature extraction using a custom VGG3D convolutional neural network
  - Standard feature extraction from the entire network
  - Stage-specific feature extraction from targeted layers (e.g., layer 20) for optimal attention on important structures
- Multiple segmentation types (0-10) and alpha blending options
- UMAP and t-SNE dimensionality reduction for feature visualization
- K-means clustering for synaptic structure classification
- Comprehensive visualization tools including:
  - 2D/3D cluster visualizations
  - Interactive plots with Plotly
  - GIF visualization of 3D synapse volumes
- Automatic report generation with HTML-based comprehensive reports
- Presynapse connectivity analysis

## Feature Extraction Methods

The system supports two methods for feature extraction:

1. **Standard Method**: Extracts features from the entire network after the last convolutional layer
2. **Stage-Specific Method**: Extracts features from a specific layer (like layer 20), which can capture the most informative features for certain structures

To specify the feature extraction method, use the following command-line arguments:
```
--extraction_method standard|stage_specific  # Choose feature extraction method
--layer_num 20                              # Layer to extract features from when using stage_specific
```

## Segmentation Types

The system supports 11 different segmentation types (0-10):

0. Raw data
1. Presynapse
2. Postsynapse
3. Both sides
4. Vesicles + Cleft (closest only)
5. Closest vesicles/cleft + sides
6. Vesicle cloud (closest)
7. Cleft (closest)
8. Mitochondria (closest)
9. Vesicle + Cleft
10. Cleft + Pre

## Project Structure

```
SynapseClusterEM/
├── synapse/                      # Main package directory
│   ├── __init__.py               # Package initialization and exports
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   └── vgg3d.py              # VGG3D model implementation
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataloader.py         # Data loading utilities
│   │   └── dataset.py            # Dataset classes
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   └── clusterhelper.py      # Clustering utilities
│   └── visualization/            # Visualization tools
│       ├── __init__.py
│       └── sample_fig.py         # Functions for creating visualizations
├── assets/                       # Application assets (logo, icons)
├── inference.py                  # Feature extraction and analysis script
├── presynapse_analysis.py        # Presynapse connectivity analysis
├── report_generator.py           # HTML report generation
├── synapse_gui.py                # Graphical user interface
└── requirements.txt              # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alim98/SynapseClusterEM.git
cd SynapseClusterEM
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data in the appropriate directory structure:
```
data/
├── raw/                  # Raw EM volumes
│   ├── bbox1/
│   ├── bbox2/
│   └── ...
├── seg/                  # Segmentation volumes
│   ├── bbox1/
│   ├── bbox2/
│   └── ...
└── excel/                # Synapse information
    └── synapse_info.xlsx
```


## Usage

### Enhanced Pipeline (v4)

For advanced analysis with improved automation and organization, use the latest pipeline version:

```bash
python run_synapse_pipeline4.py
```

The enhanced pipeline v4 offers these improvements:
- **Timestamp-based Results**: All outputs are organized in a timestamped directory structure
- **Comprehensive Logging**: Detailed logs capturing all pipeline stages and results
- **Feature Extraction Options**: 
  - Choose between standard or stage-specific feature extraction methods
  - Select specific network layers for feature extraction
  - Configure pooling methods (avg, max, concat_avg_max, spp)
- **Vesicle Analysis Integration**: Automated analysis of vesicle cloud sizes and their relationship to clusters

#### Pipeline Options

```bash
# Run pipeline with stage-specific feature extraction from layer 20
python run_synapse_pipeline4.py --extraction_method stage_specific --layer_num 20

# Run pipeline with specific pooling method
python run_synapse_pipeline4.py --pooling_method concat_avg_max

# Run only vesicle analysis on existing results
python run_synapse_pipeline4.py --only_vesicle_analysis
```

#### Data and Model Configuration

The pipeline uses specific directories for data and model checkpoints. You can customize these paths using the following arguments:

```bash
# Specify individual data directories
python run_synapse_pipeline4.py --raw_base_dir data/7_bboxes_plus_seg/raw --seg_base_dir data/7_bboxes_plus_seg/seg --add_mask_base_dir data/vesicle_cloud__syn_interface__mitochondria_annotation --excel_file data/7_bboxes_plus_seg

# Specify custom model checkpoint
python run_synapse_pipeline4.py --checkpoint_path /path/to/model.checkpoint
```

Default data directory structure:
```
data/
├── 7_bboxes_plus_seg/raw/      # Raw EM volumes
├── 7_bboxes_plus_seg/seg/      # Segmentation volumes
├── 7_bboxes_plus_seg/          # Contains Excel files with synapse information
└── vesicle_cloud__syn_interface__mitochondria_annotation/  # Additional mask data
```

Default model checkpoint:
```
hemibrain_production.checkpoint  # Pre-trained model weights file in project root
```

#### Output Structure

The pipeline automatically creates timestamped output directories:

```
results/
├── run_YYYYMMDD_HHMMSS/    # Timestamped parent directory for this run
│   ├── pipeline_log.txt    # Complete log of the pipeline execution
│   ├── csv_outputs/        # CSV files with extracted features and analysis results
│   ├── clustering_results/ # Clustering results and metrics
│   ├── gifs/               # 3D visualization GIFs
│   └── reports/            # Generated HTML reports
```

### Command Line Usage

#### Basic Analysis

Run the main analysis script for feature extraction and visualization:

```bash
python inference.py
```

#### Clustering Analysis

Perform clustering on extracted features:

```bash
python Clustering.py
```

#### Attention Visualization

Visualize model attention using Grad-CAM to understand what regions of the synapse the model focuses on:

```bash
# Basic attention visualization for a specific sample
python attention_visualization.py --sample_idx 0 --output_dir results/attention_maps

# Visualize attention with a specific layer
python attention_visualization.py --sample_idx 0 --output_dir results/attention_maps --target_layer features.20

# Multi-layer attention visualization
# This generates a comparative view of attention at early, middle, and late network layers
python attention_visualization.py --sample_idx 0 --output_dir results/multi_layer_analysis --multi_layer --n_slices 3

# Batch processing for multiple samples
python attention_visualization.py --batch_mode --n_samples 5 --output_dir results/batch_attention

# Find top attended regions across samples
python attention_visualization.py --find_top_regions --n_samples 10 --n_top_regions 5 --output_dir results/top_attention
```

The attention visualization offers these key options:
- `--sample_idx`: Index of the sample to visualize
- `--output_dir`: Directory to save attention maps
- `--target_layer`: Layer to use for Grad-CAM (e.g., features.3, features.20, features.27)
- `--multi_layer`: Compare attention across multiple network layers simultaneously
- `--n_slices`: Number of depth slices to visualize
- `--batch_mode`: Process multiple samples sequentially
- `--find_top_regions`: Identify and visualize regions with highest attention across samples

The attention maps show how the 3D model processes spatial information at different depths and network layers, revealing the progressive abstraction of synapse features through the network.

### Enhanced Grad-CAM Visualization with Consistent Grayscale Values

For the most accurate visualization of attention maps with consistent grayscale values across depth slices, use the optimized multi-layer script:

```bash
python multi_layer_cam.py
```

This script offers several advantages:
- **Consistent Grayscale Values**: Implements proper normalization at multiple stages to ensure uniform grayscale values across all slices
- **Multi-Layer Analysis**: Shows attention maps from early, middle, and late layers side by side
- **3D Attention Visualization**: Clearly demonstrates how attention patterns evolve across depth slices
- **High-Quality Output**: Generates a comprehensive visualization with proper colorbars and normalization

The multi_layer_cam.py script can be easily modified to:
- Change the sample index to visualize different samples
- Select different network layers for attention analysis
- Adjust the number of displayed slices
- Customize the visualization layout

When comparing attention across different layers in a 3D CNN, this script helps reveal how:
1. Early layers (e.g., features.3) focus on detailed local features
2. Middle layers (e.g., features.20) show more complex pattern recognition
3. Late layers (e.g., features.27) capture high-level abstractions

### Configuration

Configure the analysis by editing parameters in `synapse/utils/config.py` or passing command line arguments:

```bash
python inference.py --bbox_name bbox1 bbox2 --segmentation_type 10 --alpha 0.5
```

## Technical Details

### Feature Extraction

Features are extracted using a VGG3D neural network architecture, modified to process 3D volumes. The network includes:

- 3D convolutional layers arranged in blocks
- Max pooling for spatial reduction
- Feature extraction from intermediate layers

### Clustering

The system uses KMeans clustering to group synapses based on their structural features. Optimal cluster numbers are determined using silhouette scores and elbow method analysis.

### Dimensionality Reduction

For visualization purposes, high-dimensional features are projected to 2D/3D space using:
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

## Visualization Outputs

The analysis produces various visualization outputs:
- t-SNE plots colored by bounding box
- t-SNE plots colored by cluster assignment
- 3D interactive visualizations
- Representative sample images from each cluster
- GIF animations of 3D synapse volumes
- Comprehensive HTML reports with interactive elements
- Grad-CAM attention maps showing model focus areas at different network depths
  - Single-layer attention maps showing what features the model focuses on
  - Multi-layer comparative visualizations showing how attention evolves through the network
  - Top attended regions across multiple samples to identify common patterns


### Graphical User Interface (GUI)

The easiest way to use SynapseClusterEM is through its graphical interface:

```bash
python run_gui.py
```

The GUI provides:
- Configuration of all analysis parameters
- Selection of segmentation types (0-10) and alpha values
- Running the full pipeline or individual components
- Viewing and managing generated reports
- Saving and loading configurations
- About section with information about the Max Planck Institute for Brain Research

## Reports

Two types of reports are generated:
1. **Comprehensive Report**: Contains detailed analysis of each segmentation type and alpha combination, including visualizations, feature statistics, and clustering results.
2. **Presynapse Summary**: Focuses on presynapse connectivity analysis, showing relationships between synapses that share the same presynapse ID.

Reports are saved in:
- `results/comprehensive_reports/report_[timestamp]/index.html`
- `results/comprehensive_reports/presynapse_summary_[timestamp]/presynapse_summary.html`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was conducted at the **Max Planck Institute for Brain Research** in Frankfurt, Germany. We gratefully acknowledge the support and resources provided by the institute.

## Development Team

- **Ali Mikaeili** - Intern - Max Planck Institute for Brain Research, Frankfurt am Main
- **Ali Karimi** - Postdo - Max Planck Institute for Brain Research, Frankfurt am Main
- **Dominic Evans** - Postdoc - Max Planck Institute for Brain Research, Frankfurt am Main

## Contact

Ali Mikaeili - Mikaeili.Barzili@gmail.com
- GitHub: [@alim98](https://github.com/alim98)

## Citation

If you use this code or find it helpful for your research, please consider citing:

```
@software{MPIBRNeuralSystemsDepartment2025synapseclusterem,
  author = {Mikaeili, Ali},
  title = {SynapseClusterEM: A Framework for 3D Synapse Analysis},
  year = {2025},
  publisher = {GitHub},
  institution = {Max Planck Institute for Brain Research},
  url = {https://github.com/alim98/SynapseClusterEM}
}
```