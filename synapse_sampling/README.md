# Connectome Synapse Sampling Integration

This module integrates connectome-based synapse sampling into the existing synapse analysis pipeline. Instead of relying on local image files for synapse data, the pipeline can now directly sample synapses from the connectome.

## Components

- `synapse_sampling.py`: Core functionality to sample synapses from the connectome.
- `adapter.py`: Adapter classes to integrate connectome sampling with the existing pipeline.
- `run_synapse_pipeline_with_sampling.py`: Modified pipeline script that supports connectome sampling.

## Important Notes on Compatibility

### NumPy Version Compatibility

If you encounter errors related to NumPy version compatibility, such as:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.5 as it may crash.
```

You should downgrade NumPy to a 1.x version. You can do this with:

```bash
pip install numpy<2.0.0
```

Or in your conda environment:

```bash
conda install numpy=1.24.0
```

### Dummy Data Mode

If the original connectome paths are not available (which is likely in most environments), the sampling module will automatically use a dummy data mode that generates synthetic data. This allows for testing and development without access to the actual connectome data.

## How to Use

### Basic Usage

To run the pipeline with connectome sampling, use the `run_synapse_pipeline_with_sampling.py` script with the `--use_connectome` flag:

```bash
python run_synapse_pipeline_with_sampling.py --use_connectome
```

### Advanced Options

You can configure the sampling process with several parameters:

- `--batch_size`: Number of samples to load from the connectome (default: 10)
- `--policy`: Sampling policy - "random" for actual data, "dummy" for test data (default: "random")
- `--verbose`: Enable verbose logging during sampling

For example:

```bash
python run_synapse_pipeline_with_sampling.py --use_connectome --batch_size 20 --policy random --verbose
```

### Using Dummy Data Explicitly

If you want to explicitly use dummy data (without attempting to access the connectome), use:

```bash
python run_synapse_pipeline_with_sampling.py --use_connectome --policy dummy --batch_size 20
```

### Pipeline Parameters

You can still use the standard pipeline parameters:

- `--segmentation_type`: Segmentation type to use
- `--alpha`: Alpha value for segmentation
- `--extraction_method`: Method to extract features ("standard" or "stage_specific")
- `--layer_num`: Layer number to extract features from when using stage_specific method
- `--pooling_method`: Method to use for pooling ("avg", "max", "concat_avg_max", "spp")

## Implementation Details

The integration works by:

1. `SynapseConnectomeAdapter`: Adapts the connectome sampling to the pipeline data format
2. `ConnectomeDataset`: Provides a PyTorch Dataset interface for the sampled connectome data
3. `run_pipeline_with_connectome()`: Custom pipeline function to handle connectome data

The adapter ensures that the connectome data structure matches what the existing pipeline expects, allowing seamless integration without modifying the core pipeline code.

## Notes

- The implemented approach is similar to segmentation type 10 (presynapse + cleft mask)
- When real data is unavailable, we generate realistic looking dummy data with:
  - Random intensity values for raw volumes
  - Spherical masks to simulate presynaptic structures
- The module handles all necessary masks combining 