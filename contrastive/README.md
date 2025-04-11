# Contrastive Learning for Synapse Analysis

This module implements contrastive learning for synapse feature extraction and representation learning. The system trains a VGG3D-based contrastive model on synapse volumes, enabling effective feature representation for downstream analysis.

## Overview

The contrastive learning pipeline enables:
- Training a model to learn discriminative features from synapse volumes
- Extracting features from trained models for downstream analysis
- Configurable training parameters for optimization

## Usage

### Basic Usage

Run the complete pipeline (data processing, training, and feature extraction):

```bash
python contrastive/run_contrastive_pipeline.py --warmup_epochs 5 --gradual_epochs 5 --epochs 10
```

### Feature Extraction Only

Extract features using a pre-trained model:

```bash
python contrastive/run_contrastive_pipeline.py --extract_only --checkpoint ./contrastive/checkpoints/final_contrastive_model.pt
```

### Training Only

Train the model without feature extraction:

```bash
python contrastive/run_contrastive_pipeline.py --train_only --warmup_epochs 5 --gradual_epochs 5 --epochs 10
```

## Components

### ContrastivePipeline

The main class that coordinates the entire process:
- Data loading and preprocessing
- 3D Volume processing
- Model training with contrastive loss
- Feature extraction

### VGG3DContrastive

A VGG-based 3D CNN model modified for contrastive learning:
- Extracts features from 3D volumes
- Projects features to a lower-dimensional embedding space
- Supports contrastive learning objectives

### Synapse3DProcessor

Processes 3D synapse volumes:
- Normalizes data
- Handles different input formats
- Prepares data for model input

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--warmup_epochs` | Number of warmup training epochs | 5 |
| `--gradual_epochs` | Number of gradual training epochs | 5 |
| `--epochs` | Total number of training epochs | 20 |
| `--extract_only` | Only extract features, no training | False |
| `--train_only` | Only train the model, no feature extraction | False |
| `--checkpoint` | Path to model checkpoint for feature extraction | None |
| `--gradient_accumulation_steps` | Number of steps for gradient accumulation | 16 |

## Training Process

The training process consists of three phases:
1. **Warmup Phase**: Initial training with base learning rate
2. **Gradual Phase**: Training with gradually increasing batch size
3. **Main Phase**: Full training with optimized parameters

The system implements gradient accumulation to effectively train with larger batch sizes on memory-constrained hardware.

## Output

The pipeline produces:
- Trained model checkpoints (saved to `./contrastive/checkpoints/`)
- Feature vectors for each synapse (saved as CSV to `./contrastive/results/`)
- Training logs and metrics

## Feature Extraction

The extracted features are high-dimensional vectors that represent the learned embeddings of the synapses. These features can be used for:
- Clustering similar synapses
- Classification tasks
- Visualizations (e.g., UMAP projections)
- Other downstream analyses

## Directory Structure

- `contrastive/`
  - `data/`: Contains data loading and augmentation code
    - `augmentations.py`: 3D augmentations for contrastive learning
    - `dataset.py`: ContrastiveDataset for loading paired augmented views
  - `models/`: Contains model definitions and loss functions
    - `contrastive_model.py`: VGG3D with projection head
    - `losses.py`: NT-Xent loss implementation
  - `utils/`: Utility functions and configuration
    - `config.py`: Configuration for contrastive learning
  - `train_contrastive.py`: Script for training the contrastive model
  - `run_contrastive_pipeline.py`: Main script to run the entire pipeline

## Configuration

You can configure the contrastive learning pipeline by modifying `contrastive/utils/config.py` or by passing command line arguments:

```bash
python -m contrastive.run_contrastive_pipeline --batch_size 16 --learning_rate 0.0001 --num_epochs 50
```

### Key Configuration Parameters

- **Training parameters**:
  - `batch_size`: Batch size for training (default: 32)
  - `num_epochs`: Number of training epochs (default: 100)
  - `learning_rate`: Learning rate (default: 0.0001)
  - `weight_decay`: Weight decay for regularization (default: 1e-4)
  - `temperature`: Temperature parameter for NT-Xent loss (default: 0.07)
  - `proj_dim`: Projection head output dimension (default: 128)

- **Augmentation parameters**:
  - `rotation_range`: Maximum rotation angle in degrees (default: 15)
  - `flip_prob`: Probability of random flip (default: 0.5)
  - `noise_level`: Maximum noise level (default: 0.05)
  - `contrast_range`: Range for contrast adjustment (default: 0.2)
  - `brightness_range`: Range for brightness adjustment (default: 0.2)
  - `gaussian_blur_sigma`: Sigma for Gaussian blur (default: 1.0)

- **Model parameters**:
  - `freeze_backbone`: Whether to freeze the backbone during training (default: False)
  - `stage_to_finetune`: Which stage to fine-tune, -1 means all (default: -1)

## Algorithm Details

### NT-Xent Loss

The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss is implemented in `contrastive/models/losses.py`. This loss function:

1. Takes two sets of embeddings from the same batch (two augmented views per synapse)
2. Computes the similarity between all possible pairs using cosine similarity
3. Treats embeddings from the same synapse as positive pairs
4. Treats embeddings from different synapses as negative pairs
5. Uses a temperature parameter to scale the similarities
6. Applies cross-entropy loss to maximize the similarity of positive pairs and minimize the similarity of negative pairs

### Data Augmentations

The 3D augmentations implemented in `contrastive/data/augmentations.py` include:

- Random 3D rotations
- Random flips along each axis
- Gaussian noise addition
- Contrast adjustment
- Brightness adjustment
- Gaussian blur

## References

- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- MoCo: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
- BYOL: [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733) 