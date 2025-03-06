# Synapse Analysis Project

A deep learning and computer vision framework for analyzing and clustering 3D synapse data.

## Features

- 3D synapse image data processing
- Feature extraction using a VGG3D neural network
- UMAP and t-SNE dimensionality reduction
- Clustering analysis
- Comprehensive visualization tools

## Project Structure

```
synapse2/
├── config.py            # Configuration management
├── dataloader.py        # Data loading and processing
├── dataset.py           # PyTorch dataset classes
├── vgg3d.py             # 3D VGG neural network model
├── clusterhelper.py     # Clustering utilities
├── inference2.py        # Main inference and analysis script
├── requirements.txt     # Project dependencies
└── data/                # Data directory (not tracked)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/synapse2.git
cd synapse2
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data in the appropriate directory structure:
```
data/
├── 7_bboxes_plus_seg/
│   ├── raw/
│   │   ├── bbox1/
│   │   ├── bbox2/
│   │   └── ...
│   ├── seg/
│   │   ├── bbox1/
│   │   ├── bbox2/
│   │   └── ...
│   └── bbox1.xlsx, bbox2.xlsx, ...
└── vesicle_cloud__syn_interface__mitochondria_annotation/
    ├── bbox_1/
    ├── bbox_2/
    └── ...
```

## Usage

Run the main analysis script:

```bash
python inference2.py
```

This will:
1. Load synapse data
2. Extract features using the VGG3D model
3. Perform clustering and dimensionality reduction
4. Generate visualizations

## Visualization Outputs

The analysis produces various visualization outputs:
- t-SNE plots colored by bounding box
- t-SNE plots colored by cluster
- 3D interactive visualizations
- Sample images from each cluster

## License

[Your License]

## Acknowledgements

[Your Acknowledgements] 