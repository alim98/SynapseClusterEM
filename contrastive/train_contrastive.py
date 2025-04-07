"""
Script for training a contrastive learning model on synapse data.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import datetime
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from synapse pipeline
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from synapse.models import Vgg3D, load_model_from_checkpoint
from synapse import config as synapse_config
from synapse_pipeline import SynapsePipeline
from vgg3d_stage_extractor import VGG3DStageExtractor

# Import contrastive modules
from contrastive.utils.config import config as contrastive_config
from contrastive.models.contrastive_model import VGG3DContrastive, initialize_contrastive_model
from contrastive.models.losses import NTXentLoss, SimplifiedNTXentLoss
from contrastive.data.dataset import ContrastiveDataset, create_contrastive_dataloader
from contrastive.data.augmentations import ContrastiveAugmenter, ToTensor3D


def setup_logging(config):
    """
    Set up logging for the training process.
    
    Args:
        config: Configuration object
        
    Returns:
        logging.Logger: Logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"train_contrastive_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_data(config):
    """
    Load data for contrastive learning.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (vol_data_dict, syn_df, processor)
    """
    # Create a synapse pipeline to load data
    pipeline = SynapsePipeline(config)
    
    # Load and prepare data - this returns (dataset, dataloader)
    pipeline.load_data()
    
    # Access the vol_data_dict and syn_df directly from the pipeline
    vol_data_dict = pipeline.vol_data_dict
    syn_df = pipeline.syn_df
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    processor.normalize_volume = True
    
    return vol_data_dict, syn_df, processor


def create_optimizer(model, config):
    """
    Create optimizer for contrastive learning.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Define parameters to optimize
    params = model.parameters()
    
    # Create optimizer
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    # Create learning rate scheduler
    if config.use_scheduler:
        if config.scheduler_type.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=1e-6
            )
        elif config.scheduler_type.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif config.scheduler_type.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {config.scheduler_type}")
    else:
        scheduler = None
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, epoch, loss, config, filename=None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        config: Configuration object
        filename: Optional filename override
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename if not provided
    if filename is None:
        filename = f"contrastive_checkpoint_epoch{epoch:03d}.pt"
    
    # Prepend checkpoint directory
    filepath = os.path.join(config.checkpoint_dir, filename)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {k: v for k, v in vars(config).items() if not k.startswith("_")}
    }, filepath)
    
    # Also save just the backbone for easy loading in other scripts
    backbone_filename = f"backbone_epoch{epoch:03d}.pt"
    backbone_filepath = os.path.join(config.checkpoint_dir, backbone_filename)
    torch.save(model.backbone.state_dict(), backbone_filepath)
    
    return filepath


def load_checkpoint(model, optimizer, config, filename):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        config: Configuration object
        filename: Checkpoint filename
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    # Prepend checkpoint directory if not already included
    if not os.path.dirname(filename):
        filename = os.path.join(config.checkpoint_dir, filename)
    
    # Load checkpoint
    checkpoint = torch.load(filename)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch and loss
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return model, optimizer, epoch, loss


def train_contrastive(model, dataloader, optimizer, scheduler, criterion, device, config, logger):
    """
    Train the contrastive model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader with contrastive data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        config: Configuration object
        logger: Logger instance
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    
    # Initialize metrics
    running_loss = 0.0
    n_batches = 0
    
    # Training loop
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        if batch is None:
            continue  # Skip empty batches
        
        # Unpack batch
        (view1, view2), syn_infos, bbox_names = batch
        
        # Move data to device
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Calculate loss
        loss = criterion(z1, z2)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        n_batches += 1
        
        # Log batch info
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    # Update scheduler if using
    if scheduler is not None:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(running_loss / n_batches)
        else:
            scheduler.step()
    
    # Calculate average loss
    avg_loss = running_loss / n_batches if n_batches > 0 else float('inf')
    
    return avg_loss


def validate_contrastive(model, dataloader, criterion, device, logger):
    """
    Validate the contrastive model.
    
    Args:
        model: Model to validate
        dataloader: DataLoader with validation data
        criterion: Loss function
        device: Device to validate on
        logger: Logger instance
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    
    # Initialize metrics
    running_loss = 0.0
    n_batches = 0
    
    # Validation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            if batch is None:
                continue  # Skip empty batches
            
            # Unpack batch
            (view1, view2), syn_infos, bbox_names = batch
            
            # Move data to device
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            # Forward pass
            z1 = model(view1)
            z2 = model(view2)
            
            # Calculate loss
            loss = criterion(z1, z2)
            
            # Update metrics
            running_loss += loss.item()
            n_batches += 1
    
    # Calculate average loss
    avg_loss = running_loss / n_batches if n_batches > 0 else float('inf')
    
    return avg_loss


def main():
    """Main function for contrastive learning training."""
    # Get configuration
    config = contrastive_config
    config.parse_args()
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Starting contrastive learning training with config: {vars(config)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    vol_data_dict, syn_df, processor = load_data(config)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloader = create_contrastive_dataloader(vol_data_dict, syn_df, processor, config, batch_size=config.batch_size)
    
    # Initialize model
    logger.info("Initializing model...")
    model = initialize_contrastive_model(config)
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)
    
    # Initialize loss function
    criterion = NTXentLoss(temperature=config.temperature, device=device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if hasattr(config, 'resume_from') and config.resume_from:
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, config, config.resume_from)
        start_epoch += 1  # Start from the next epoch
    
    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {config.num_epochs}")
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        
        # Train for one epoch
        train_loss = train_contrastive(model, dataloader, optimizer, scheduler, criterion, device, config, logger)
        
        # Log epoch results
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        # Save checkpoint if improved
        if train_loss < best_loss:
            best_loss = train_loss
            logger.info(f"New best loss: {best_loss:.6f}")
            save_checkpoint(model, optimizer, epoch, train_loss, config, "best_contrastive_model.pt")
        
        # Save regular checkpoint
        if epoch % config.save_every == 0 or epoch == config.num_epochs - 1:
            save_checkpoint(model, optimizer, epoch, train_loss, config)
    
    # Save final model
    logger.info("Training complete. Saving final model.")
    save_checkpoint(model, optimizer, config.num_epochs - 1, train_loss, config, "final_contrastive_model.pt")
    
    logger.info("Contrastive training finished successfully.")


if __name__ == "__main__":
    main() 