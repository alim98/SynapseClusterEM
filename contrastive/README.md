# Contrastive Learning for Synapse Analysis

This module implements contrastive learning for fine-tuning the VGG3D model on synapse data. The contrastive learning approach learns more robust representations by comparing different augmented views of the same synapse volume.

## Overview

The contrastive learning approach works by:

1. Taking a synapse volume
2. Creating two different augmented views of the same volume (e.g., rotations, flips, contrast changes)
3. Passing both views through the VGG3D model and a projection head
4. Training the model to pull representations of the same synapse closer together while pushing representations of different synapses apart
5. Using the fine-tuned VGG3D backbone for feature extraction

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

## Usage

### Running the Full Pipeline

To run the entire contrastive learning pipeline (data loading, model training, feature extraction):

```bash
python -m contrastive.run_contrastive_pipeline
```

### Training Only

To only train the contrastive model:

```bash
python -m contrastive.run_contrastive_pipeline --train_only
```

### Feature Extraction Only

To extract features from a trained model:

```bash
python -m contrastive.run_contrastive_pipeline --extract_only --checkpoint path/to/checkpoint.pt --layer_num 20
```

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