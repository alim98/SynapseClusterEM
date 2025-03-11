# Synapse Analysis Pipeline - Complete Reference

## Pipeline Overview

The synapse analysis pipeline processes 3D electron microscopy data to analyze synaptic structures. The pipeline consists of the following main stages:

1. **Data Loading**: Load raw EM volumes, segmentation masks, and additional masks
2. **Model Loading**: Load the VGG3D deep learning model
3. **Feature Extraction**: Extract features from the model (standard or stage-specific methods)
4. **Clustering**: Cluster the extracted features to identify synapse types
5. **Visualization**: Create UMAP/t-SNE visualizations colored by different attributes
6. **Presynapse Analysis**: Analyze presynaptic regions and their relationships
7. **Vesicle Analysis**: Analyze vesicle cloud sizes and their relationship to clusters
8. **Attention Map Analysis**: Generate and analyze attention maps for different clusters

## Installation and Setup

The pipeline requires the following Python packages:

```bash
python -m pip install statsmodels scikit-image plotly pandas numpy umap-learn
```

## Usage

### Basic Usage

To run the full pipeline with the default configuration:

```bash
python run_synapse_pipeline.py
```

### Configuration Options

The pipeline uses the configuration system from `synapse/config.py` and extends it with additional parameters for feature extraction and vesicle analysis. You can set parameters as follows:

```bash
python run_synapse_pipeline.py --segmentation_type 1 --alpha 1.0 
```

#### Pipeline-Specific Options

| Option | Type | Description |
|--------|------|-------------|
| `--only_vesicle_analysis` | flag | Only run vesicle size analysis on existing results |
| `--extraction_method` | string | Feature extraction method ('standard' or 'stage_specific') |
| `--layer_num` | integer | Layer number to extract features from when using stage_specific method |

#### Pipeline Control Flags

| Flag | Description |
|---------------|-------------|
| `--skip_feature_extraction` | Skip feature extraction stage (use existing features) |
| `--skip_clustering` | Skip clustering stage (use existing clusters) |
| `--skip_visualization` | Skip visualization stage |
| `--skip_presynapse_analysis` | Skip presynapse analysis stage |

### Example Commands

Run only feature extraction and clustering:
```bash
python run_synapse_pipeline.py --skip_visualization --skip_presynapse_analysis
```

Run only vesicle analysis on existing results:
```bash
python run_synapse_pipeline.py --only_vesicle_analysis
```

Run with stage-specific feature extraction from layer 20:
```bash
python run_synapse_pipeline.py --extraction_method stage_specific --layer_num 20
```

## Output Directory Structure

The pipeline creates the following directory structure for outputs:

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

## Required Output Deliverables and Procedures

### 1. Data Samples from Bounding Boxes with Mask Overlay

#### Checklist:
- [ ] Representative image samples showing bounding boxes with overlaid segmentation masks
- [ ] Multiple examples across different image regions
- [ ] Clear visualization of how masks align with the underlying structures

#### Implementation Procedure:
**Functions Used:**
- `SynapseDataset` and `SynapseDataLoader` classes from the `synapse` module
- `sample_fig.py` for visualization

**Steps:**
1. Load raw EM volumes using `SynapseDataLoader`
2. Load segmentation and additional masks
3. Process data samples using `Synapse3DProcessor`
4. Overlay segmentation masks on raw data
5. Generate visualization samples showing:
   - Raw EM data
   - Segmentation masks
   - Overlaid view of masks on raw data
6. Save these visualizations to the output directory

The code in `sample_fig.py` handles this visualization process, creating multi-panel figures that show how the masks align with the original structures in the EM data.

### 2. Clustered Feature Data

#### Checklist:
- [ ] CSV file containing extracted features clustered by similarity
- [ ] Columns should include:
  - [ ] Bounding box/object identifiers
  - [ ] Extracted feature values
  - [ ] Assigned cluster labels
  - [ ] Relevant metadata (coordinates, size metrics, etc.)

#### Implementation Procedure:
**Functions Used:**
- `extract_features` or `extract_stage_specific_features` in `inference.py`
- `run_clustering_analysis` in `inference.py`
- `Clustering.py` for clustering algorithms

**Steps:**
1. Features are extracted from either:
   - Standard method: Final layer of VGG3D (192 features)
   - Stage-specific method: Targeted intermediate layer like layer 20 (96 features)
2. Feature data is saved in a CSV file with columns:
   - Bounding box identifiers
   - Extracted feature values (feat_1 to feat_192 or layer20_feat_1 to layer20_feat_96)
   - Metadata like coordinates and size metrics
3. Clustering is performed using methods from `Clustering.py`
4. Cluster assignments are added to the CSV
5. The final clustered CSV is saved to `clustering_results_seg{seg_type}_alpha{alpha}/clustered_features.csv`

### 3. Dimensionality Reduction Visualizations

#### Checklist:
- [ ] UMAP visualizations (2D):
  - [ ] Version 1: Points colored by bounding box/object identifiers
  - [ ] Version 2: Points colored by cluster assignment
- [ ] t-SNE visualizations (2D):
  - [ ] Version 1: Points colored by bounding box/object identifiers
  - [ ] Version 2: Points colored by cluster assignment
- [ ] Legends and scales for all visualizations
- [ ] Explanatory captions describing feature distribution patterns

#### Implementation Procedure:
**Functions Used:**
- `apply_umap` in `inference.py`
- `create_bbox_colored_umap` and `create_cluster_colored_umap` in `presynapse_analysis.py`
- `create_umap_with_vesicle_sizes` in `vesicle_size_visualizer.py`

**Steps:**
1. UMAP dimensionality reduction:
   - Load clustered features from CSV
   - Apply UMAP algorithm to reduce high-dimensional features to 2D space
   - Generate two versions of UMAP plots:
     - Version 1: Points colored by bounding box/object identifiers
     - Version 2: Points colored by cluster assignment
   - Add interactive tooltips, legends, and scales
   - Save as interactive HTML files

2. t-SNE dimensionality reduction:
   - Similar procedure as UMAP but using t-SNE algorithm
   - Generate two visualizations with different coloring schemes
   - Save as interactive HTML files

These visualizations help analyze how synapses cluster in feature space and how well those clusters correspond to known structures.

### 4. Presynapse Analysis Report

#### Checklist:
- [ ] Statistical summary of presynaptic structures
- [ ] Distribution metrics (count, density, spatial distribution)
- [ ] Morphological characteristics
- [ ] Comparison to expected values/ranges (if applicable)
- [ ] Visualizations of key metrics

#### Implementation Procedure:
**Functions Used:**
- `identify_synapses_with_same_presynapse` in `presynapse_analysis.py`
- `calculate_feature_distances` in `presynapse_analysis.py`
- `analyze_cluster_membership` in `presynapse_analysis.py`
- `generate_report` in `presynapse_analysis.py`
- `run_presynapse_analysis` orchestrates the entire process

**Steps:**
1. Load feature data and segmentation volumes
2. Identify synapses that share the same presynaptic region using segmentation IDs
3. Calculate feature distances between synapses within and across presynapses
4. Analyze how synapses from the same presynapse distribute across clusters
5. Generate distance heatmaps and connected UMAP visualizations
6. Create statistical summaries of presynaptic structures:
   - Distribution metrics (count, density, spatial distribution)
   - Morphological characteristics
   - Distance comparisons between related synapses
7. Generate a comprehensive HTML report with all findings

This analysis helps understand how presynaptic regions relate to the feature-based clustering, testing whether synapses from the same presynaptic structure are more similar.

### 5. Vesicle Size Analysis Report

#### Checklist:
- [ ] Size distribution histogram/plots
- [ ] Summary statistics (mean, median, std dev, etc.)
- [ ] Comparison across different clusters or regions
- [ ] Correlation with other measured features
- [ ] Anomaly detection (unusually large/small vesicles)

#### Implementation Procedure:
**Functions Used:**
- `compute_vesicle_cloud_sizes` in `vesicle_size_visualizer.py`
- `analyze_vesicle_sizes_by_cluster` in `vesicle_size_visualizer.py`
- `count_bboxes_in_clusters` and `plot_bboxes_in_clusters` in `vesicle_size_visualizer.py`

**Steps:**
1. Calculate vesicle cloud sizes for each synapse:
   - Extract the closest connected component to the center coordinate
   - Calculate volume, surface area, and other metrics
2. Merge vesicle size data with feature data and cluster assignments
3. Generate visualizations:
   - Size distribution histograms/plots per cluster
   - Box plots and violin plots comparing sizes across clusters
   - UMAP visualizations with point sizes representing vesicle sizes
4. Perform statistical analysis:
   - Calculate summary statistics (mean, median, standard deviation)
   - Run ANOVA tests to compare sizes across clusters
   - Calculate effect sizes for significant differences
5. Generate cumulative distribution functions and probability density functions
6. Create an HTML report with all visualizations and findings

This analysis examines how vesicle cloud sizes relate to the feature-based clusters, potentially identifying morphological subtypes of synapses.

### 6. Attention Map Samples per Cluster

#### Checklist:
- [ ] Representative samples from each identified cluster
- [ ] Attention maps highlighting critical features that define each cluster
- [ ] Side-by-side comparison of original image and attention visualization
- [ ] Explanatory text describing what features are being highlighted

#### Implementation Procedure:
**Functions Used:**
- `SimpleGradCAM` class in `multi_layer_cam.py`
- `process_single_sample` in `multi_layer_cam.py`
- `visualize_cluster_attention` in `multi_layer_cam.py`
- `AttentionMaskAnalyzer` class in `attention_mask_analyzer.py`
- `create_cluster_sample_visualizations` in `synapse_pipeline.py`

**Steps:**
1. Find representative samples from each cluster (typically 4 samples per cluster)
2. For each sample:
   - Run it through the VGG3D model
   - Generate Grad-CAM attention maps at multiple layers (especially layer 20)
   - Overlay attention maps on original EM data
   - Create visualizations showing which parts of the synapse the model is attending to
3. Generate side-by-side comparisons:
   - Original EM data
   - Attention map overlay
   - Segmentation mask
4. Create a grid of sample visualizations for each cluster
5. Add explanatory text describing the distinctive features of each cluster

This analysis helps understand what features the model is using to distinguish between different synapse types, providing interpretability for the clustering results.

## Visualization Outputs

The pipeline generates the following detailed visualizations:

### 1. UMAP Visualizations
- UMAP colored by bounding box
- UMAP colored by cluster
- UMAP with vesicle sizes as point sizes

### 2. Cluster Sample Visualizations
- 4 sample images from each cluster
- Attention maps for each sample

### 3. Vesicle Analysis Visualizations
- Box plots of vesicle cloud sizes by cluster
- Violin plots of vesicle distributions
- Statistical summary tables
- Effect size comparison matrices
- Cumulative distribution functions
- Probability density function visualizations

### 4. Bounding Box Analysis
- Bar charts of bounding box counts in each cluster
- Box plots of bounding box distributions
- Scatter plots showing bounding box vs. cluster relationships
- Pie charts of bounding box proportions

## Pipeline Components

The pipeline consists of the following main files:

- **synapse_pipeline.py**: The main high-level orchestrator class
- **run_synapse_pipeline.py**: Command line interface for running the pipeline
- **vesicle_size_visualizer.py**: Functions for analyzing and visualizing vesicle cloud sizes
- **inference.py**: Core functions for model inference and feature extraction
- **multi_layer_cam.py**: Implementation of attention mapping using Grad-CAM
- **presynapse_analysis.py**: Functions for analyzing presynaptic regions
- **Clustering.py**: Implementations of clustering algorithms
- **sample_fig.py**: Functions for creating sample visualizations
- **attention_mask_analyzer.py**: Functions for analyzing attention masks

## Technical Details

The pipeline uses several key technologies:

1. **Model**: VGG3D deep learning model for feature extraction
2. **Feature Extraction**: Either standard (final layer) or stage-specific (intermediate layer)
3. **Dimensionality Reduction**: UMAP and t-SNE for visualizing high-dimensional features
4. **Clustering**: Hierarchical clustering (default) with options for K-means and other methods
5. **Visualization**: Interactive Plotly-based visualizations for exploring the data
6. **Attention Mapping**: Grad-CAM for visualizing model attention
7. **Statistical Analysis**: ANOVA, t-tests, and effect size calculations for quantitative comparisons

## Customization

To customize the default parameters of the pipeline, modify the values in `pipeline_config.py`. This allows you to change default values for:

- Feature extraction parameters
- Clustering parameters
- Visualization options
- Output paths

## Additional Notes

- All visualizations should include appropriate axes, legends, and scale bars
- Reports should include methodology descriptions
- Where applicable, include statistical significance assessments
- Consider including a summary dashboard that combines key insights from all analysis components 