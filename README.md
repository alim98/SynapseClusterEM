# SynapseClusterEM

A deep learning framework for analyzing and clustering 3D synapse structures from electron microscopy (EM) data.

## Overview

SynapseClusterEM is a powerful tool designed to analyze 3D synapse morphology using advanced neural networks and unsupervised learning techniques. The project aims to identify structural patterns and classify synapses based on their 3D architecture, providing insights into brain connectivity and function.

## Features

- 3D synapse image data processing from electron microscopy datasets
- Feature extraction using a custom VGG3D convolutional neural network
- Multiple segmentation types and alpha blending options
- UMAP and t-SNE dimensionality reduction for feature visualization
- K-means clustering for synaptic structure classification
- Comprehensive visualization tools including:
  - 2D/3D cluster visualizations
  - Interactive plots with Plotly
  - GIF visualization of 3D synapse volumes

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
├── run_analysis.py               # Main entry point script
├── Clustering.py                 # Clustering analysis script
├── inference.py                  # Feature extraction script
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

### Graphical User Interface (GUI)

The easiest way to use SynapseClusterEM is through its graphical interface:

```bash
python synapse_gui.py
```

The GUI provides:
- Configuration of all analysis parameters
- Running the full pipeline or individual components
- Viewing and managing generated reports
- Saving and loading configurations

![GUI Screenshot](docs/images/gui_screenshot.png)

### Command Line Usage

#### Basic Analysis

Run the main analysis script for feature extraction and visualization:

```bash
python inference.py
```

### Clustering Analysis

Perform clustering on extracted features:

```bash
python Clustering.py
```

### Configuration

Configure the analysis by editing parameters in `synapse/utils/config.py` or passing command line arguments:

```bash
python run_analysis.py --bbox_name bbox1 bbox2 --segmentation_type 9 --alpha 0.5
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 MPIBR Neural Systems Department  


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgements

This work was conducted during an internship at the **Max Planck Institute for Brain Research**. I would like to express my gratitude to:

- My supervisors and mentors at Max Planck Institute for their guidance and support
- Neural Systems Department for providing resources and expertise

## Contact

Ali Mikaeili - Mikaeili.Barzili@gmail.com
- GitHub: [@alim98](https://github.com/alim98)
- Affiliation: University of Tehran, M.Sc. in Artificial Intelligence and Robotics
- Intern at Max Planck Institute for Brain Research

## Citation

If you use this code or find it helpful for your research, please consider citing:

```
@software{MPIBRNeuralSystemsDepartment2025synapseclusterem,
  author = {Mikaeili, Ali},
  title = {SynapseClusterEM: A Framework for 3D Synapse Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alim98/SynapseClusterEM}
}
```