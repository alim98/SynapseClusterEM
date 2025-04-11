"""
Main script for running the contrastive learning synapse analysis pipeline.

This script demonstrates how to use contrastive learning to fine-tune
the VGG3D model for synapse analysis.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from original synapse pipeline
from synapse import config as synapse_config
from synapse.utils import SynapseConfig
from synapse_pipeline import SynapsePipeline
from synapse.models import Vgg3D, load_model_from_checkpoint
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from vgg3d_stage_extractor import VGG3DStageExtractor

# Import contrastive modules
from contrastive.utils.config import ContrastiveConfig
from contrastive.models.contrastive_model import VGG3DContrastive, initialize_contrastive_model
from contrastive.models.losses import NTXentLoss, SimplifiedNTXentLoss
from contrastive.data.dataset import ContrastiveDataset, create_contrastive_dataloader
from contrastive.data.augmentations import ContrastiveAugmenter, ToTensor3D
from contrastive.train_contrastive import train_contrastive, save_checkpoint, load_checkpoint


def create_optimizer_and_scheduler(model, config):
    """
    Create optimizer and scheduler for the model.
    
    Args:
        model (nn.Module): The model
        config (Config): The configuration
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': model.projection.parameters(),
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
    optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    return optimizer, scheduler


def setup_logging(config):
    """Set up logging for the contrastive pipeline."""
    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"contrastive_pipeline_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Contrastive learning pipeline started")
    
    return logger


def log_print(logger, *args, **kwargs):
    """Print and log a message."""
    message = " ".join(str(arg) for arg in args)
    logger.info(message)
    print(*args, **kwargs)


class ContrastivePipeline:
    """Pipeline for contrastive learning on synapse data."""
    
    def __init__(self, config=None, synapse_config=None):
        """Initialize the contrastive learning pipeline."""
        # Set up configurations
        self.config = config or ContrastiveConfig(
            batch_size=2,  # Reduced batch size from 4 to 2
            gradient_accumulation_steps=16,  # Increased from 8 to 16 to compensate
            learning_rate=1e-4,
            weight_decay=1e-6,
            temperature=0.07,
            num_epochs=50,
            warmup_epochs=5,
            gradual_epochs=5,
            results_dir='contrastive/results',
            checkpoint_dir='contrastive/checkpoints',
            log_dir='contrastive/logs',
            visualization_dir='contrastive/visualization'
        )
        
        self.synapse_config = synapse_config or SynapseConfig()
        
        # Create necessary directories
        os.makedirs(self.config.results_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.visualization_dir, exist_ok=True)
        
        # Initialize logger
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'contrastive.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize other components as None
        self.model = None
        self.processor = None
        self.train_loader = None
        self.val_loader = None
        self.visualizer = None
        self.base_model = None
        
        # Set memory optimization settings
        if torch.cuda.is_available():
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Set memory allocation settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            self.logger.info("Memory optimization settings applied")
    
    def load_and_prepare_data(self):
        """Load and prepare data for contrastive learning."""
        self.logger.info("Loading and preparing data...")
        
        # Create a synapse pipeline to load data
        pipeline = SynapsePipeline(self.synapse_config)
        
        # Load and prepare data
        pipeline.load_data()
        
        # Access the vol_data_dict and syn_df directly from the pipeline
        self.vol_data_dict = pipeline.vol_data_dict
        self.syn_df = pipeline.syn_df
        
        # Initialize processor
        self.processor = Synapse3DProcessor(size=self.synapse_config.size)
        self.processor.normalize_volume = True
        
        # Create data loaders
        self.train_loader = create_contrastive_dataloader(
            self.vol_data_dict, self.syn_df, self.processor, 
            self.config, batch_size=self.config.batch_size
        )
        
        # Use the same loader for validation for now
        self.val_loader = self.train_loader
        
        self.logger.info(f"Loaded {len(self.syn_df)} synapse samples across {len(self.vol_data_dict)} volumes")
    
    def setup_model(self):
        """Set up the contrastive model."""
        self.logger.info("Setting up contrastive model...")
        
        # Initialize contrastive model directly
        self.model = initialize_contrastive_model(self.config)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'backbone'):
            # Enable gradient checkpointing for the backbone
            if hasattr(self.model.backbone, 'features'):
                # Apply gradient checkpointing to each sequential module in features
                for i, module in enumerate(self.model.backbone.features):
                    if hasattr(module, 'use_checkpointing'):
                        module.use_checkpointing = True
                        self.logger.info(f"Enabled gradient checkpointing in backbone features module {i}")
            
            # Enable gradient checkpointing for the projection head if it exists
            if hasattr(self.model, 'projection'):
                if hasattr(self.model.projection, 'use_checkpointing'):
                    self.model.projection.use_checkpointing = True
                    self.logger.info("Enabled gradient checkpointing in projection head")
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")
        
        return self.model
    
    def setup_base_model(self):
        """Set up the base VGG3D model for comparison."""
        self.logger.info("Setting up base VGG3D model...")
        self.base_model = Vgg3D()
        self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        return self.base_model
    
    def extract_base_features(self, checkpoint_path=None):
        """Extract features using the base VGG3D model."""
        self.logger.info("Extracting features from base model...")
        
        if self.base_model is None:
            self.setup_base_model()
            
        if checkpoint_path:
            self.logger.info(f"Loading base model checkpoint: {checkpoint_path}")
            self.base_model.load_state_dict(torch.load(checkpoint_path))
        
        features_list = []
        
        with torch.no_grad():
            for batch_idx, (views, synapse_info, bbox_names) in enumerate(self.train_loader):
                view1s, _ = views
                view1s = view1s.to(self.device)
                
                # Get features from base model
                features = self.base_model(view1s)
                
                # Convert to numpy
                features_np = features.cpu().numpy()
                
                # Create DataFrame for this batch
                for i, (syn_info, bbox_name) in enumerate(zip(synapse_info, bbox_names)):
                    feature_dict = {
                        'bbox_name': bbox_name,
                        **syn_info
                    }
                    
                    # Add features
                    for j in range(features_np[i].shape[0]):
                        feature_dict[f'feature_{j}'] = features_np[i][j]
                    
                    features_list.append(feature_dict)
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed batch {batch_idx}/{len(self.train_loader)}")
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save features
        features_path = os.path.join(self.config.results_dir, "base_features.csv")
        features_df.to_csv(features_path, index=False)
        self.logger.info(f"Saved base features to {features_path}")
        
        return features_df
    
    def train_model(self):
        """Train the contrastive model."""
        self.logger.info("Training contrastive model...")
        
        # Load data if not already loaded
        if self.train_loader is None:
            self.load_and_prepare_data()
        
        # Set up model if not already set up
        if self.model is None:
            self.setup_model()
        
        # Create criterion
        criterion = NTXentLoss(
            temperature=self.config.temperature,
            device=self.device
        ).to(self.device)
        
        # Training phases
        phases = [
            ('warmup', self.config.warmup_epochs),
            ('gradual', self.config.gradual_epochs),
            ('full', self.config.epochs - self.config.warmup_epochs - self.config.gradual_epochs)
        ]
        
        # Training loop
        for phase, num_epochs in phases:
            self.logger.info(f"Starting {phase} training phase for {num_epochs} epochs")
            
            # Create optimizer and scheduler for this phase
            optimizer, scheduler = create_optimizer_and_scheduler(self.model, self.config)
            
            # Training loop
            for epoch in range(num_epochs):
                # Train
                train_loss = train_contrastive(
                    model=self.model,
                    train_loader=self.train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    config=self.config,
                    logger=self.logger
                )
                
                # Log epoch results
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_interval == 0:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"contrastive_model_{phase}_epoch_{epoch+1}.pt"
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'train_loss': train_loss,
                        'phase': phase
                    }, checkpoint_path)
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(self.config.checkpoint_dir, "final_contrastive_model.pt")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Saved final model to {final_path}")
    
    def extract_features(self, checkpoint_path=None):
        """
        Extract features from the model.
        
        Args:
            checkpoint_path: Path to checkpoint to load
            
        Returns:
            DataFrame with extracted features
        """
        self.logger.info("Extracting features...")
        
        # Load data if not already loaded
        if self.train_loader is None:
            self.load_and_prepare_data()
        
        # Set up model if not already set up
        if self.model is None:
            self.setup_model()
        
        # Load checkpoint if specified
        if checkpoint_path:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
        
        # Extract features
        self.model.eval()
        features_list = []
        
        with torch.no_grad():
            for batch_idx, (views, synapse_info, bbox_names) in enumerate(self.train_loader):
                # Move views to device
                view1s, view2s = views
                view1s = view1s.to(self.device)
                
                # Get features
                # Check if the model returns features when return_features=True
                if hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames:
                    _, features = self.model(view1s, return_features=True)
                else:
                    # If the model doesn't support return_features, use the first view's projections as features
                    features = self.model(view1s)
                
                # Convert to numpy
                features_np = features.cpu().numpy()
                
                # Create DataFrame for this batch
                for i, (syn_info, bbox_name) in enumerate(zip(synapse_info, bbox_names)):
                    feature_dict = {
                        'bbox_name': bbox_name,
                        **syn_info
                    }
                    
                    # Add features
                    for j in range(features_np[i].shape[0]):
                        feature_dict[f'feature_{j}'] = features_np[i][j]
                    
                    features_list.append(feature_dict)
                
                # Log progress
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed batch {batch_idx}/{len(self.train_loader)}")
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save features to CSV
        features_path = os.path.join(self.config.results_dir, "contrastive_features.csv")
        features_df.to_csv(features_path, index=False)
        self.logger.info(f"Saved features to {features_path}")
        
        return features_df
    
    def visualize_features(self, features_df, base_features_df=None):
        """Visualize features using UMAP and clustering."""
        self.logger.info("Visualizing features...")
        
        # Create visualization directory
        vis_dir = os.path.join(self.config.visualization_dir, "gif_umap")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create GIF UMAP visualization
        from contrastive.visualization.gif_umap.GifUmapContrastive import GifUmapContrastive
        gif_umap = GifUmapContrastive(
            features_df=features_df,
            output_dir=vis_dir,
            method_name="Contrastive Learning"
        )
        gif_umap.create_visualization()
        
        # Create standard UMAP visualization
        if self.visualizer is None:
            from contrastive.visualization.visualize import FeatureVisualizer
            self.visualizer = FeatureVisualizer(
                self.config.visualization_dir,
                self.logger
            )
        
        # Analyze contrastive features
        contrastive_results = self.visualizer.analyze_features(
            features_df,
            save_dir=os.path.join(self.config.visualization_dir, "contrastive")
        )
        
        # Compare with base model if available
        if base_features_df is not None:
            self.logger.info("Comparing with base model features...")
            contrastive_features, _ = self.visualizer.extract_feature_columns(features_df)
            base_features, _ = self.visualizer.extract_feature_columns(base_features_df)
            
            self.visualizer.compare_models(
                base_features,
                contrastive_features,
                labels=contrastive_results['labels'],
                save_dir=os.path.join(self.config.visualization_dir, "comparison")
            )
        
        return contrastive_results
    
    def run_pipeline(self):
        """Run the full contrastive learning pipeline."""
        self.logger.info("Running full contrastive learning pipeline...")
        
        # Load data
        self.load_and_prepare_data()
        
        # Set up models
        self.setup_model()
        self.setup_base_model()
        
        # Extract base features
        base_features_df = self.extract_base_features()
        
        # Train model if no checkpoint exists
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "final_contrastive_model.pt")
        if not os.path.exists(checkpoint_path):
            self.logger.info("No checkpoint found. Starting training...")
            self.train_model()
        else:
            self.logger.info(f"Found checkpoint at {checkpoint_path}")
        
        # Extract features
        features_df = self.extract_features(checkpoint_path)
        
        # Visualize features
        self.visualize_features(features_df, base_features_df)
        
        self.logger.info("Pipeline completed successfully")
        return features_df


def main():
    """Main function to run the contrastive learning pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run contrastive learning pipeline")
    parser.add_argument("--train_only", action="store_true", help="Only run training phase")
    parser.add_argument("--extract_only", action="store_true", help="Only run feature extraction")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for feature extraction")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--gradual_epochs", type=int, default=5, help="Number of gradual unfreezing epochs")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs")
    args = parser.parse_args()
    
    # Create configurations
    contrastive_config = ContrastiveConfig()
    synapse_config = SynapseConfig()
    
    # Update config with command line arguments
    contrastive_config.warmup_epochs = args.warmup_epochs
    contrastive_config.gradual_epochs = args.gradual_epochs
    contrastive_config.epochs = args.epochs
    
    # Initialize pipeline
    pipeline = ContrastivePipeline(contrastive_config, synapse_config)
    
    try:
        if args.extract_only:
            if not args.checkpoint:
                raise ValueError("--checkpoint must be specified when using --extract_only")
            pipeline.logger.info("Running feature extraction only...")
            features_df = pipeline.extract_features(args.checkpoint)
            pipeline.logger.info("Feature extraction completed successfully")
            return features_df
        
        if args.train_only:
            pipeline.logger.info("Running training phase only...")
            pipeline.train_model()
            pipeline.logger.info("Training completed successfully")
            return
        
        # Run full pipeline
        pipeline.logger.info("Running full pipeline...")
        features_df = pipeline.run_pipeline()
        pipeline.logger.info("Pipeline completed successfully")
        return features_df
    
    except Exception as e:
        pipeline.logger.error(f"Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main() 