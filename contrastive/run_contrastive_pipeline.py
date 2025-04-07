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
from synapse_pipeline import SynapsePipeline
from synapse.models import Vgg3D, load_model_from_checkpoint
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from vgg3d_stage_extractor import VGG3DStageExtractor

# Import contrastive modules
from contrastive.utils.config import config as contrastive_config
from contrastive.models.contrastive_model import VGG3DContrastive, initialize_contrastive_model
from contrastive.models.losses import NTXentLoss, SimplifiedNTXentLoss
from contrastive.data.dataset import ContrastiveDataset, create_contrastive_dataloader
from contrastive.data.augmentations import ContrastiveAugmenter, ToTensor3D
from contrastive.train_contrastive import train_contrastive, save_checkpoint, load_checkpoint


def setup_logging():
    """Set up logging for the contrastive pipeline."""
    # Create log directory if it doesn't exist
    os.makedirs(contrastive_config.log_dir, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(contrastive_config.log_dir, f"contrastive_pipeline_{timestamp}.log")
    
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
    """
    Pipeline for contrastive learning and feature extraction on synapse data.
    """
    def __init__(self, config=None, synapse_config=None):
        """
        Initialize the contrastive pipeline.
        
        Args:
            config: Contrastive learning configuration
            synapse_config: Synapse pipeline configuration
        """
        self.config = config or contrastive_config
        self.synapse_config = synapse_config or synapse_config
        
        # Initialize logger
        self.logger = setup_logging()
        
        # Initialize state
        self.vol_data_dict = None
        self.syn_df = None
        self.model = None
        self.contrastive_model = None
        self.dataloader = None
        self.processor = None
        
        # Create output directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        log_print(self.logger, "ContrastivePipeline initialized")
    
    def load_and_prepare_data(self):
        """
        Load and prepare data for contrastive learning.
        
        Returns:
            tuple: (vol_data_dict, syn_df)
        """
        log_print(self.logger, "Loading and preparing data...")
        
        # Create a synapse pipeline to load data
        pipeline = SynapsePipeline(self.synapse_config)
        
        # Load data using the synapse pipeline - this returns (dataset, dataloader)
        pipeline.load_data()
        
        # Now access the vol_data_dict and syn_df directly from the pipeline
        self.vol_data_dict = pipeline.vol_data_dict
        self.syn_df = pipeline.syn_df
        
        log_print(self.logger, f"Loaded {len(self.syn_df)} synapse samples across {len(self.vol_data_dict)} volumes")
        
        return self.vol_data_dict, self.syn_df
    
    def setup_model(self):
        """
        Set up the contrastive learning model.
        
        Returns:
            VGG3DContrastive: The contrastive model
        """
        log_print(self.logger, "Setting up contrastive model...")
        
        # Initialize the contrastive model
        self.contrastive_model = initialize_contrastive_model(self.config)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_model = self.contrastive_model.to(device)
        
        log_print(self.logger, f"Model set up on {device}")
        
        return self.contrastive_model
    
    def setup_dataloader(self):
        """
        Set up the data loader for contrastive learning.
        
        Returns:
            DataLoader: The contrastive data loader
        """
        log_print(self.logger, "Setting up data loader...")
        
        # Load data if not already loaded
        if self.vol_data_dict is None or self.syn_df is None:
            self.load_and_prepare_data()
        
        # Initialize processor
        self.processor = Synapse3DProcessor(size=self.config.size)
        self.processor.normalize_volume = True
        
        # Create contrastive data loader
        self.dataloader = create_contrastive_dataloader(
            self.vol_data_dict,
            self.syn_df,
            self.processor,
            self.config,
            batch_size=self.config.batch_size
        )
        
        log_print(self.logger, f"Data loader created with batch size {self.config.batch_size}")
        
        return self.dataloader
    
    def train_model(self):
        """
        Train the contrastive model.
        
        Returns:
            str: Path to the trained model checkpoint
        """
        log_print(self.logger, "Starting contrastive model training...")
        
        # Set up model if not already set up
        if self.contrastive_model is None:
            self.setup_model()
        
        # Set up data loader if not already set up
        if self.dataloader is None:
            self.setup_dataloader()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize optimizer
        from torch.optim import Adam, SGD
        if self.config.optimizer.lower() == "adam":
            optimizer = Adam(
                self.contrastive_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = SGD(
                self.contrastive_model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        
        # Initialize scheduler
        if self.config.use_scheduler:
            if self.config.scheduler_type.lower() == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.num_epochs,
                    eta_min=1e-6
                )
            elif self.config.scheduler_type.lower() == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=30,
                    gamma=0.1
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=10,
                    verbose=True
                )
        else:
            scheduler = None
        
        # Initialize loss function
        criterion = NTXentLoss(temperature=self.config.temperature, device=device)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_loss = float('inf')
        if hasattr(self.config, 'resume_from') and self.config.resume_from:
            log_print(self.logger, f"Resuming from checkpoint: {self.config.resume_from}")
            self.contrastive_model, optimizer, start_epoch, best_loss = load_checkpoint(
                self.contrastive_model, optimizer, self.config, self.config.resume_from
            )
            start_epoch += 1  # Start from the next epoch
        
        # Training loop
        log_print(self.logger, f"Starting training from epoch {start_epoch} to {self.config.num_epochs}")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            log_print(self.logger, f"Epoch {epoch}/{self.config.num_epochs}")
            
            # Train for one epoch
            train_loss = train_contrastive(
                self.contrastive_model,
                self.dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                self.config,
                self.logger
            )
            
            # Log epoch results
            log_print(self.logger, f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
            
            # Save checkpoint if improved
            if train_loss < best_loss:
                best_loss = train_loss
                log_print(self.logger, f"New best loss: {best_loss:.6f}")
                checkpoint_path = save_checkpoint(
                    self.contrastive_model,
                    optimizer,
                    epoch,
                    train_loss,
                    self.config,
                    "best_contrastive_model.pt"
                )
            
            # Save regular checkpoint
            if epoch % self.config.save_every == 0 or epoch == self.config.num_epochs - 1:
                checkpoint_path = save_checkpoint(
                    self.contrastive_model,
                    optimizer,
                    epoch,
                    train_loss,
                    self.config
                )
        
        # Save final model
        log_print(self.logger, "Training complete. Saving final model.")
        final_checkpoint_path = save_checkpoint(
            self.contrastive_model,
            optimizer,
            self.config.num_epochs - 1,
            train_loss,
            self.config,
            "final_contrastive_model.pt"
        )
        
        log_print(self.logger, f"Final model saved to {final_checkpoint_path}")
        
        return final_checkpoint_path
    
    def extract_features(self, checkpoint_path=None, layer_num=None):
        """
        Extract features from the trained contrastive model.
        
        Args:
            checkpoint_path (str, optional): Path to the model checkpoint
            layer_num (int, optional): Layer number to extract features from
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        log_print(self.logger, "Extracting features from contrastive model...")
        
        # Load model if not already loaded
        if self.contrastive_model is None:
            self.setup_model()
        
        # Load checkpoint if specified
        if checkpoint_path is not None:
            log_print(self.logger, f"Loading checkpoint from {checkpoint_path}")
            # Just load the model weights, not optimizer
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                self.contrastive_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.contrastive_model.load_state_dict(checkpoint)
        
        # Load data if not already loaded
        if self.vol_data_dict is None or self.syn_df is None:
            self.load_and_prepare_data()
        
        # Initialize processor if not already initialized
        if self.processor is None:
            self.processor = Synapse3DProcessor(size=self.config.size)
            self.processor.normalize_volume = True
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_model = self.contrastive_model.to(device)
        self.contrastive_model.eval()
        
        # Initialize dataset for feature extraction (using standard SynapseDataset, not ContrastiveDataset)
        dataset = SynapseDataset(
            vol_data_dict=self.vol_data_dict,
            synapse_df=self.syn_df,
            processor=self.processor,
            segmentation_type=self.config.segmentation_type,
            subvol_size=self.config.subvol_size,
            num_frames=self.config.num_frames,
            alpha=self.config.alpha,
            normalize_across_volume=True,
        )
        
        # Extract features for each sample
        features_list = []
        synapse_ids = []
        bbox_names = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Extracting features"):
                sample = dataset[idx]
                if sample is None:
                    continue
                
                pixel_values, syn_info, bbox_name = sample
                
                # Skip invalid samples
                if pixel_values is None:
                    continue
                
                # Move to device
                pixel_values = pixel_values.unsqueeze(0).to(device)  # Add batch dimension
                
                # Extract features
                if layer_num is not None:
                    # Extract from specific layer
                    features = self.contrastive_model.extract_features(pixel_values, layer_num=layer_num)
                else:
                    # Extract from backbone before projection head
                    features, _ = self.contrastive_model(pixel_values, return_features=True)
                
                # Convert to numpy and flatten
                features_np = features.cpu().numpy().flatten()
                
                # Add to lists
                features_list.append(features_np)
                synapse_ids.append(syn_info['Var1'])
                bbox_names.append(bbox_name)
        
        # Create DataFrame with features
        features_df = pd.DataFrame(features_list)
        
        # Add metadata
        features_df['Var1'] = synapse_ids
        features_df['bbox_name'] = bbox_names
        
        # Save features to CSV
        output_filename = f"contrastive_features_layer{layer_num if layer_num is not None else 'backbone'}.csv"
        output_path = os.path.join(self.config.results_dir, output_filename)
        features_df.to_csv(output_path, index=False)
        
        log_print(self.logger, f"Extracted features for {len(features_df)} samples, saved to {output_path}")
        
        return features_df
    
    def run_pipeline(self):
        """
        Run the entire contrastive learning pipeline.
        """
        log_print(self.logger, "Starting contrastive learning pipeline...")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Set up model
        self.setup_model()
        
        # Step 3: Set up data loader
        self.setup_dataloader()
        
        # Step 4: Train model
        checkpoint_path = self.train_model()
        
        # Step 5: Extract features
        features_df = self.extract_features(checkpoint_path)
        
        # Step 6: Extract features from specific layer (layer 20 is typically used)
        layer20_features_df = self.extract_features(checkpoint_path, layer_num=20)
        
        log_print(self.logger, "Contrastive learning pipeline completed successfully!")
        
        return {
            'checkpoint_path': checkpoint_path,
            'features_df': features_df,
            'layer20_features_df': layer20_features_df
        }


def main():
    """Main function for running the contrastive learning pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Contrastive Learning Pipeline")
    parser.add_argument("--train_only", action="store_true", help="Only train the contrastive model")
    parser.add_argument("--extract_only", action="store_true", help="Only extract features from a trained model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for feature extraction")
    parser.add_argument("--layer_num", type=int, default=20, help="Layer number to extract features from")
    
    args, _ = parser.parse_known_args()
    
    # Parse config arguments
    contrastive_config.parse_args()
    synapse_config.parse_args()
    
    # Create pipeline
    pipeline = ContrastivePipeline(contrastive_config, synapse_config)
    
    if args.train_only:
        # Train the model
        pipeline.load_and_prepare_data()
        pipeline.setup_model()
        pipeline.setup_dataloader()
        checkpoint_path = pipeline.train_model()
        print(f"Training completed, model saved to {checkpoint_path}")
    
    elif args.extract_only:
        # Extract features from a trained model
        if args.checkpoint is None:
            print("Error: Must specify --checkpoint when using --extract_only")
            return
        
        pipeline.load_and_prepare_data()
        pipeline.setup_model()
        features_df = pipeline.extract_features(args.checkpoint, args.layer_num)
        print(f"Feature extraction completed for {len(features_df)} samples")
    
    else:
        # Run the entire pipeline
        results = pipeline.run_pipeline()
        print(f"Pipeline completed successfully!")
        print(f"Trained model saved to {results['checkpoint_path']}")
        print(f"Extracted features for {len(results['features_df'])} samples")


if __name__ == "__main__":
    main() 