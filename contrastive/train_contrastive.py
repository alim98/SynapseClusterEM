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


def create_optimizer_and_scheduler(model, config):
    """
    Create optimizer and scheduler with different learning rates for different parts of the model.
    
    Args:
        model (nn.Module): The model
        config (Config): The configuration
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': model.projection_head.parameters(),
            'lr': config.learning_rate
        }
    ]
    
    # Add backbone parameters with lower learning rate
    if hasattr(model, 'backbone'):
        param_groups.append({
            'params': model.backbone.parameters(),
            'lr': config.learning_rate * 0.1  # Lower learning rate for backbone
        })
    
    # Create optimizer
    optimizer = optim.Adam(param_groups, weight_decay=config.weight_decay)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
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


def train_contrastive(model, train_loader, criterion, optimizer, config, logger):
    """Train the model using contrastive learning with gradient accumulation.
    
    Args:
        model: The VGG3D model with projection head
        train_loader: DataLoader for training data
        criterion: NT-Xent loss function
        optimizer: Optimizer for model parameters
        config: Training configuration
        logger: Logger instance
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    optimizer.zero_grad()  # Zero gradients at start
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    # Track accumulated gradients
    accumulated_batches = 0
    
    for batch_idx, (views, _, _) in enumerate(train_loader):
        try:
            # Unpack views
            view1s, view2s = views
            
            # Move data to device
            view1s = view1s.to(next(model.parameters()).device)
            view2s = view2s.to(next(model.parameters()).device)
            
            # Forward pass with mixed precision
            if config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Get projections for both views
                    z_i = model(view1s)
                    z_j = model(view2s)
                    loss = criterion(z_i, z_j) / config.gradient_accumulation_steps
            else:
                # Get projections for both views
                z_i = model(view1s)
                z_j = model(view2s)
                loss = criterion(z_i, z_j) / config.gradient_accumulation_steps
            
            # Backward pass with mixed precision
            if config.use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Track loss (multiply back by accumulation steps for logging)
            total_loss += loss.item() * config.gradient_accumulation_steps
            accumulated_batches += 1
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.info(f'Batch [{batch_idx}/{num_batches}], Current Loss: {loss.item() * config.gradient_accumulation_steps:.4f}')
            
            # Update weights if we've accumulated enough gradients
            if accumulated_batches == config.gradient_accumulation_steps or batch_idx == num_batches - 1:
                # Clip gradients
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Reset accumulation counter
                accumulated_batches = 0
                
                # Log accumulated step
                logger.info(f'Performed optimization step after {batch_idx + 1} batches')
                
                # Clear cache periodically
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clear unnecessary tensors
            del view1s, view2s, z_i, z_j, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error(f"WARNING: out of memory at batch {batch_idx}. Skipping batch.")
                continue
            else:
                raise e
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches
    logger.info(f'Average Loss: {avg_loss:.4f}')
    
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


def train_model(model, train_loader, val_loader, config, logger):
    """
    Train the model with gradual unfreezing.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): The training dataloader
        val_loader (DataLoader): The validation dataloader
        config (Config): The configuration
        logger (Logger): The logger
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create criterion
    criterion = NTXentLoss(
        temperature=config.temperature,
        batch_size=config.batch_size
    ).to(device)
    
    # Training phases
    phases = [
        ('warmup', config.warmup_epochs),
        ('gradual', config.gradual_epochs),
        ('full', config.epochs - config.warmup_epochs - config.gradual_epochs)
    ]
    
    for phase, num_epochs in phases:
        logger.info(f"Starting {phase} training phase for {num_epochs} epochs")
        
        # Set training phase
        model.set_training_phase(phase)
        
        # Create optimizer and scheduler for this phase
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss = train_contrastive(
                model, train_loader, criterion, optimizer, config, logger
            )
            
            # Validate
            val_loss = validate_contrastive(
                model, val_loader, criterion, device, logger
            )
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % config.save_interval == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f"contrastive_model_{phase}_epoch_{epoch+1}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'phase': phase
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "final_contrastive_model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")


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
    
    # Train model
    train_model(model, dataloader, dataloader, config, logger)
    
    logger.info("Contrastive training finished successfully.")


if __name__ == "__main__":
    main() 