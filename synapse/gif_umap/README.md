# GifUmap.py Documentation

## Overview

`GifUmap.py` is a powerful visualization and clustering tool for 3D volumetric data. It provides functionality to cluster feature vectors extracted from neural network models and visualize them using dimensionality reduction techniques (UMAP or t-SNE) with interactive elements like animated GIFs.

## Key Features

- **Clustering Analysis**: Performs k-means clustering on feature vectors
- **Dimensionality Reduction**: Supports both UMAP and t-SNE visualization methods
- **Multiple Visualization Types**:
  - Interactive Plotly visualizations with hover effects
  - Matplotlib static visualizations with GIF thumbnails
  - Seaborn visualizations with grid views
  - Custom HTML visualizations with embedded animated GIFs
  - Animated GIF visualization with draggable elements
- **Balanced Sampling**: Ensures equal representation from each cluster when selecting samples for GIFs
- **Auto-looping GIFs**: Creates continuously looping GIFs from volumetric data

## Prerequisites

- Python 3.7+
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - umap-learn
  - plotly
  - matplotlib
  - seaborn
  - Pillow (PIL)

## Usage

### Basic Command

```bash
python GifUmap.py
```

### Command-line Arguments

- `--dim-reduction`: Choose dimensionality reduction method (`umap` or `tsne`, default is `umap`)
- Additional arguments inherited from `synapse.config`

Example:
```bash
python GifUmap.py --dim-reduction tsne
```

## Visualization Types

### 1. Interactive Plotly Visualization

- HTML-based interactive scatter plot
- Points colored by cluster or bounding box
- Hover to see sample details
- File: `{dim_reduction}_with_gifs.html`

### 2. Embedded GIFs Visualization

- Interactive HTML with GIFs positioned at their UMAP/t-SNE coordinates
- Click on highlighted points to view GIFs
- Pan and zoom functionality
- File: `{dim_reduction}_with_gifs_at_coordinates.html`

### 3. Animated GIF Visualization

- HTML with draggable animated GIFs
- Base64-encoded GIFs embedded directly in the HTML
- Controls for showing/hiding and resizing GIFs
- File: `animated_gifs_{dim_reduction}_visualization.html`

### 4. Static Visualizations

- Matplotlib scatter plot with GIF thumbnails
- Seaborn visualization with cluster coloring
- Grid view of GIF thumbnails
- Files: 
  - `umap_with_gif_thumbnails.png`
  - `seaborn_umap_with_gifs.png`
  - `gif_thumbnails_grid.png`

## Key Functions

### `perform_clustering_analysis(config, csv_path, output_path)`

Performs clustering analysis on features from a CSV file.

- **Parameters**:
  - `config`: Configuration object with clustering parameters
  - `csv_path`: Path to CSV file with feature vectors
  - `output_path`: Directory to save results

### `create_umap_with_gifs(features_df, dataset, output_path, num_samples=10, random_seed=42, dim_reduction='umap')`

Creates a UMAP or t-SNE visualization with GIFs for selected samples.

- **Parameters**:
  - `features_df`: DataFrame with features and cluster assignments
  - `dataset`: Dataset for generating GIFs
  - `output_path`: Directory to save results
  - `num_samples`: Number of random samples to show with GIFs
  - `random_seed`: Random seed for reproducibility
  - `dim_reduction`: Dimensionality reduction method ('umap' or 'tsne')

### `create_gif_from_volume(volume, output_path, fps=10, quality=95)`

Creates an animated GIF from a 3D volume.

- **Parameters**:
  - `volume`: 3D array representing the volume data
  - `output_path`: Path to save the GIF
  - `fps`: Frames per second for the animation
  - `quality`: Quality of the GIF (1-100)

### `ensure_gif_autoplay(gif_paths, loop=0)`

Ensures GIFs are set to auto-loop by modifying their loop parameter.

- **Parameters**:
  - `gif_paths`: Dictionary mapping sample indices to GIF paths
  - `loop`: Loop parameter (0 = infinite, -1 = no loop, n = number of loops)

## Output Directory Structure

```
output_path/
├── clustered_features.csv       # CSV with cluster assignments
├── sample_gifs/                 # Directory containing GIFs
│   ├── bbox1_sample_2_2.gif
│   ├── bbox3_sample_6_6.gif
│   └── ...
├── umap_with_gifs.html          # Plotly visualization
├── umap_with_gifs_at_coordinates.html  # GIFs at coordinates
├── animated_gifs_umap_visualization.html  # Draggable GIFs
├── umap_with_gif_thumbnails.png  # Static visualization
├── seaborn_umap_with_gifs.png    # Seaborn visualization
├── gif_thumbnails_grid.png       # Grid of thumbnails
└── plotly_express_umap.html      # Simple Plotly Express
```

## Example Workflow

1. Extract features from a neural network model and save to CSV
2. Run `GifUmap.py` to perform clustering and create visualizations
3. Open the generated HTML files in a web browser to explore the data
4. Use the visualization to identify patterns and relationships in the data

## Troubleshooting

### Common Issues

- **"Error: No suitable coordinate columns found"**: Ensure your feature DataFrame has appropriate columns.
- **"Warning: Could not initialize dataset"**: Check that dataset paths in configuration are correct.
- **GIFs not visible in HTML**: Try using the embedded base64 version (`animated_gifs_visualization.html`).

## Notes

- The script automatically tries to load an existing dataset or initialize one from available sources.
- For large datasets, consider reducing `num_samples` to improve performance.
- For better visualization, ensure your feature vectors contain meaningful information.

## Future Improvements

- Additional clustering algorithms beyond k-means
- More customization options for visualizations
- Support for other data types beyond volumetric data
- Integration with additional dimensionality reduction techniques 