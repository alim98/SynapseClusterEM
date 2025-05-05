# Synapse Analysis Pipeline

This high-level orchestrator integrates with your existing synapse analysis codebase to provide a comprehensive pipeline. The pipeline leverages your existing configuration system and extends it with additional functionality:

1. **Data Loading**: Loads data using your dataloader.py and dataset.py
2. **Model Loading**: Loads the VGG3D model
3. **Feature Extraction**: Extracts features using the VGG3D model
4. **Clustering**: Clusters the extracted features
5. **Visualization**: Creates various visualizations of the results
6. **Presynapse Analysis**: Analyzes presynaptic regions
7. **Vesicle Analysis**: Analyzes vesicle cloud sizes and their relationship to clusters

## Installation

The pipeline requires the following Python packages:

```bash
python -m pip install statsmodels scikit-image plotly pandas numpy umap-learn
```

## Usage

### Basic Usage

To run the full pipeline with your existing configuration:

```bash
python run_synapse_pipeline.py
```

For the enhanced pipeline with improved logging and organization (recommended):

```bash
python run_synapse_pipeline4.py
```

### Configuration

The pipeline uses your existing configuration system from `synapse/config.py` and extends it with additional parameters for vesicle analysis. You can set standard parameters as you normally would:

```bash
python run_synapse_pipeline4.py --segmentation_type 1 --alpha 1.0 
```

Additionally, you can use pipeline-specific flags:

```bash
python run_synapse_pipeline4.py --only_vesicle_analysis
```

### Pipeline Options

The pipeline adds the following options to your existing configuration:

| Option | Type | Description |
|--------|------|-------------|
| `--only_vesicle_analysis` | flag | Only run vesicle size analysis on existing results |
| `--extraction_method` | string | Feature extraction method ('standard' or 'stage_specific') |
| `--layer_num` | integer | Layer number to extract features from when using stage_specific method |
| `--pooling_method` | string | Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp') |
| `--raw_base_dir` | string | Directory containing raw EM volumes (default: 'data/7_bboxes_plus_seg/raw') |
| `--seg_base_dir` | string | Directory containing segmentation volumes (default: 'data/7_bboxes_plus_seg/seg') |
| `--add_mask_base_dir` | string | Directory containing additional mask data (default: 'data/vesicle_cloud__syn_interface__mitochondria_annotation') |
| `--excel_file` | string | Directory containing Excel files with synapse information (default: 'data/7_bboxes_plus_seg') |
| `--checkpoint_path` | string | Path to model checkpoint (default: 'hemibrain_production.checkpoint') |

The pipeline also integrates with your existing flags:

| Existing Flag | Description |
|---------------|-------------|
| `--skip_feature_extraction` | Skip feature extraction stage (use existing features) |
| `--skip_clustering` | Skip clustering stage (use existing clusters) |
| `--skip_visualization` | Skip visualization stage |
| `--skip_presynapse_analysis` | Skip presynapse analysis stage |

### Feature Extraction Methods

The pipeline supports two different methods for feature extraction:

1. **Standard Method**: Extracts features from the entire network after the last convolutional layer (default)
2. **Stage-Specific Method**: Extracts features from a specific layer, such as layer 20, which can be more effective for capturing certain features

To use stage-specific feature extraction:

```bash
python run_synapse_pipeline4.py --extraction_method stage_specific --layer_num 20
```

Layer 20 is recommended as it has been found to provide the most attention on important structures in many cases.

### Example Commands

Run only feature extraction and clustering:
```bash
python run_synapse_pipeline4.py --skip_visualization --skip_presynapse_analysis
```

Run only vesicle analysis on existing results:
```bash
python run_synapse_pipeline4.py --only_vesicle_analysis
```

Run the pipeline with a specific segmentation type and alpha value:
```bash
python run_synapse_pipeline4.py --segmentation_type 10 --alpha 0.5
```

Use stage-specific feature extraction with max pooling:
```bash
python run_synapse_pipeline4.py --extraction_method stage_specific --layer_num 20 --pooling_method max
```

Use custom data directories and model checkpoint:
```bash
python run_synapse_pipeline4.py --raw_base_dir /path/to/raw --seg_base_dir /path/to/seg --add_mask_base_dir /path/to/masks --excel_file /path/to/excel --checkpoint_path /path/to/model.checkpoint
```

## Output Structure

The enhanced pipeline creates a timestamped directory structure for outputs:

```
results/
├── run_YYYYMMDD_HHMMSS/    # Timestamped parent directory for this run
│   ├── pipeline_log.txt    # Complete log of the pipeline execution
│   ├── csv_outputs/        # CSV files with extracted features and analysis results
│   ├── clustering_results/ # Clustering results and metrics
│   ├── gifs/               # 3D visualization GIFs
│   └── reports/            # Generated HTML reports
```

The standard pipeline creates the following directory structure for outputs:

```
results/
├── features_seg{segmentation_type}_alpha{alpha}/
│   └── features.csv
├── clustering_results_seg{segmentation_type}_alpha{alpha}/
│   └── clustered_features.csv
├── visualizations_seg{segmentation_type}_alpha{alpha}/
│   ├── umap_bbox_colored.html
│   └── umap_cluster_colored.html
├── sample_visualizations_seg{segmentation_type}_alpha{alpha}/
│   └── (sample visualizations with attention maps)
├── presynapse_analysis_seg{segmentation_type}_alpha{alpha}/
│   └── (presynapse analysis results)
└── vesicle_analysis_seg{segmentation_type}_alpha{alpha}/
    ├── umap_cluster_vesicle_size.html
    ├── umap_bbox_vesicle_size.html
    ├── vesicle_size_analysis.html
    └── bbox_cluster_analysis.html
```

## Pipeline Components

The pipeline consists of the following main files:

- **synapse_pipeline.py**: The main high-level orchestrator class
- **vesicle_size_visualizer.py**: Functions for analyzing and visualizing vesicle cloud sizes
- **run_synapse_pipeline.py**: Command line interface for running the pipeline

## Visualization Outputs

The pipeline generates the following visualizations:

1. **UMAP Visualizations**:
   - UMAP colored by bounding box
   - UMAP colored by cluster
   - UMAP with vesicle sizes as point sizes

2. **Cluster Sample Visualizations**:
   - 4 sample images from each cluster
   - Attention maps for each sample

3. **Vesicle Analysis Visualizations**:
   - Box plots of vesicle cloud sizes by cluster
   - Violin plots of vesicle distributions
   - Statistical summary tables
   - Effect size comparison matrices
   - Cumulative distribution functions
   - Probability density function visualizations

4. **Bounding Box Analysis**:
   - Bar charts of bounding box counts in each cluster
   - Box plots of bounding box distributions
   - Scatter plots showing bounding box vs. cluster relationships
   - Pie charts of bounding box proportions

## Customization

To customize the default parameters of the pipeline, modify the values in `pipeline_config.py`. 