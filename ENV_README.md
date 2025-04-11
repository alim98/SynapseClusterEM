# Environment Setup for Synapse Contrastive Learning

This document provides instructions for setting up the Python environment required to run the contrastive learning module.

## Option 1: Using Conda/Mamba (Recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

2. Create the environment from the provided `environment.yml` file:

   ```bash
   # Using conda
   conda env create -f environment.yml

   # Using mamba (faster)
   mamba env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate synapse2
   ```

### GPU Support

The environment is configured to support CUDA for GPU acceleration. If you experience issues with the CUDA version:

1. Determine your CUDA version:
   ```bash
   nvidia-smi
   ```

2. Modify the `environment.yml` file to specify the correct CUDA version, for example:
   ```yaml
   - pytorch>=2.0.0
   - torchvision
   - pytorch-cuda=11.8  # Adjust version as needed
   ```

## Option 2: Using Pip

1. It's recommended to create a virtual environment first:

   ```bash
   # Using venv
   python -m venv synapse2_env
   
   # On Windows
   synapse2_env\Scripts\activate
   
   # On Linux/Mac
   source synapse2_env/bin/activate
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### GPU Support with Pip

For GPU support with pip, you may need to install the appropriate PyTorch version manually:

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Testing the Installation

To verify your installation, run the test command:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Running Contrastive Learning Pipeline

After setting up your environment, you can run the contrastive learning pipeline:

```bash
# Complete pipeline
python contrastive/run_contrastive_pipeline.py --warmup_epochs 5 --gradual_epochs 5 --epochs 10

# Feature extraction only
python contrastive/run_contrastive_pipeline.py --extract_only --checkpoint ./contrastive/checkpoints/final_contrastive_model.pt

# Training only
python contrastive/run_contrastive_pipeline.py --train_only --warmup_epochs 5 --gradual_epochs 5 --epochs 10
```

For more detailed information, refer to the main documentation in `contrastive/README.md`. 