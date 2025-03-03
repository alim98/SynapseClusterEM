# Using the SynapseClusterEM Main Script

The `main.py` script serves as the primary entry point for the SynapseClusterEM project, providing a comprehensive workflow for analyzing and clustering 3D synapse structures from electron microscopy (EM) data.

## Features

- **Modular Workflow**: Run the entire pipeline or specific stages (preprocessing, feature extraction, clustering, visualization)
- **Comprehensive Logging**: Detailed logs for tracking progress and debugging
- **Reproducibility**: Automatically saves all parameters for future reference
- **Flexible Clustering**: Support for multiple clustering methods (K-means, DBSCAN)
- **Advanced Visualization**: Generate 2D and 3D visualizations of synapse clusters
- **Global Normalization**: Optional global normalization for improved model performance

## Usage

### Basic Usage

To run the complete analysis pipeline:

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --output_dir outputs/my_analysis
```

### Running Specific Stages

The script supports running specific stages of the pipeline using the `--mode` parameter:

#### Preprocessing Only

Calculate global normalization statistics:

```bash
python main.py \
    --mode preprocess \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --output_dir outputs/preprocessing \
    --use_global_norm
```

#### Feature Extraction Only

Extract features using a pre-trained model:

```bash
python main.py \
    --mode extract \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --output_dir outputs/features \
    --use_global_norm \
    --global_stats_path outputs/preprocessing/global_stats.json
```

#### Clustering Only

Perform clustering on previously extracted features:

```bash
python main.py \
    --mode cluster \
    --output_dir outputs/features \
    --n_clusters 10 \
    --clustering_method kmeans
```

#### Visualization Only

Create visualizations from previously clustered data:

```bash
python main.py \
    --mode visualize \
    --output_dir outputs/features \
    --create_3d_plots \
    --save_interactive
```

## Common Use Cases

### Analyzing Multiple Bounding Boxes

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --bbox_names bbox1 bbox2 bbox3 bbox4 bbox5 bbox6 bbox7 \
    --output_dir outputs/multiple_bboxes
```

### Comparing Different Segmentation Types

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --segmentation_types 9 10 \
    --output_dir outputs/seg_comparison
```

### Finding Optimal Number of Clusters

Run multiple clustering analyses with different numbers of clusters:

```bash
for n in 5 10 15 20; do
    python main.py \
        --mode cluster \
        --output_dir outputs/cluster_optimization/k_${n} \
        --n_clusters $n
done
```

### Using GPU Acceleration

The script automatically uses GPU if available. To optimize performance:

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --batch_size 8 \
    --num_workers 4
```

## Advanced Options

### Global Normalization

Enable global normalization for improved model performance:

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --use_global_norm \
    --num_samples_for_stats 200
```

### DBSCAN Clustering

Use DBSCAN instead of K-means for clustering:

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --clustering_method dbscan
```

### Interactive Visualizations

Generate interactive HTML visualizations:

```bash
python main.py \
    --raw_base_dir /path/to/raw/data \
    --seg_base_dir /path/to/seg/data \
    --excel_dir /path/to/excel/files \
    --checkpoint_path /path/to/vgg3d_checkpoint.pth \
    --create_3d_plots \
    --save_interactive
```

## Troubleshooting

- **Memory Issues**: Reduce batch size with `--batch_size 1`
- **Slow Processing**: Increase number of workers with `--num_workers 4`
- **Missing Features**: Ensure the feature extraction step completed successfully
- **Visualization Errors**: Check if the clustering step generated valid cluster assignments 