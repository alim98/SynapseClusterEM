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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import umap
import seaborn as sns

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
    
    def extract_features(self, checkpoint_path=None, use_backbone_features=True, layer_idx=None):
        """
        Extract features using the contrastive model.
        
        Args:
            checkpoint_path (str, optional): Path to the checkpoint to load
            use_backbone_features (bool): Whether to use raw backbone features (True) or 
                                        projection features (False)
            layer_idx (int, optional): If provided, extract features from a specific layer
                                      of the backbone
            
        Returns:
            pd.DataFrame: DataFrame with extracted features and metadata
        """
        self.logger.info("Extracting features from contrastive model...")
        
        if self.model is None:
            self.setup_model()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.logger.info(f"Loading contrastive model checkpoint: {checkpoint_path}")
            load_checkpoint(self.model, checkpoint_path, self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create empty list to store features and metadata
        all_features = []
        all_metadata = []
        
        # Extract features for each synapse
        with torch.no_grad():
            for idx, row in tqdm(self.syn_df.iterrows(), total=len(self.syn_df)):
                try:
                    # Get synapse metadata
                    syn_id = row['synapse_id']
                    bbox_name = row['bbox_name']
                    vol_id = row['vol_id']
                    
                    # Get corresponding volume data
                    vol_data = self.vol_data_dict[vol_id]
                    
                    # Get coordinates
                    x_min, y_min, z_min = row['x_min'], row['y_min'], row['z_min']
                    x_max, y_max, z_max = row['x_max'], row['y_max'], row['z_max']
                    
                    # Extract patch using processor
                    patch = self.processor.extract_patch(
                        vol_data, (x_min, y_min, z_min), (x_max, y_max, z_max)
                    )
                    
                    # Convert to tensor and add batch dimension
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
                    patch_tensor = patch_tensor.to(self.device)
                    
                    # Choose feature extraction method based on parameters
                    if use_backbone_features:
                        # Extract features from backbone
                        if layer_idx is not None:
                            # Extract features from a specific layer
                            features = self.model.extract_features_from_layer(patch_tensor, layer_idx)
                else:
                            # Use the full backbone features
                            features = self.model.backbone.features(patch_tensor)
                            
                        # Flatten features
                        features = features.view(features.size(0), -1)
                        
                        # Log feature shape before flattening for reference
                        if idx == 0:
                            self.logger.info(f"Original feature shape (before flattening): {features.shape}")
                    else:
                        # Use projection features (lower dimensional)
                        backbone_features, projection_features = self.model(patch_tensor, return_features=True)
                        features = projection_features
                    
                    # Convert to numpy and add to list
                    features_np = features.cpu().numpy().squeeze()
                    
                    # Create metadata dictionary
                    metadata = {
                        'synapse_id': syn_id,
                        'bbox_name': bbox_name,
                        'vol_id': vol_id,
                        'x_min': x_min,
                        'y_min': y_min,
                        'z_min': z_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'z_max': z_max
                    }
                    
                    # Add to lists
                    all_features.append(features_np)
                    all_metadata.append(metadata)
                    
                except Exception as e:
                    self.logger.error(f"Error extracting features for synapse {idx}: {e}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_metadata)
        
        # Add feature columns
        features_array = np.array(all_features)
        
        # Check for high dimensionality and warn
        if features_array.shape[1] > 1000:
            self.logger.warning(f"High-dimensional features detected: {features_array.shape[1]} dimensions")
            self.logger.info("Consider using PCA or UMAP for visualization")
        
        # Add feature columns to DataFrame
        for i in range(features_array.shape[1]):
            features_df[f'feature_{i}'] = features_array[:, i]
        
        # Save features to CSV
        output_path = os.path.join(self.config.results_dir, 'extracted_features.csv')
        features_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved features to {output_path}")
        
        # Return features
        return features_df
    
    def extract_features_from_layer(self, x, layer_idx):
        """
        Extract features from a specific layer in the backbone.
        Helper method to be added to VGG3DContrastive class.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int): Index of the layer to extract features from
            
        Returns:
            torch.Tensor: Features from the specified layer
        """
        # Add this method to the VGG3DContrastive class
        if not hasattr(self.model, 'extract_features_from_layer'):
            self.logger.info("Adding extract_features_from_layer method to model")
            
            def _extract_features_from_layer(self_model, x, layer_idx):
                """Extract features from a specific layer in the backbone."""
                if not hasattr(self_model, 'backbone') or not hasattr(self_model.backbone, 'features'):
                    raise ValueError("Model doesn't have backbone.features attribute")
                
                # Extract features up to the specified layer
                features = x
                for i, layer in enumerate(self_model.backbone.features):
                    features = layer(features)
                    if i == layer_idx:
                        return features
                
                # If layer_idx is out of range, return the last layer's output
                return features
            
            # Add method to the model
            import types
            self.model.extract_features_from_layer = types.MethodType(_extract_features_from_layer, self.model)
    
    def visualize_features(self, features_df, base_features_df=None, use_pca=True, n_components=50, 
                          save_prefix='contrastive', plot_3d=False):
        """
        Visualize features using dimensionality reduction techniques.
        
        Args:
            features_df (pd.DataFrame): DataFrame with extracted features
            base_features_df (pd.DataFrame, optional): DataFrame with base model features for comparison
            use_pca (bool): Whether to use PCA as preprocessing step for high-dimensional features
            n_components (int): Number of PCA components to use for preprocessing
            save_prefix (str): Prefix for saved visualization files
            plot_3d (bool): Whether to create 3D plots in addition to 2D
        """
        self.logger.info("Visualizing features...")
        
        # Create output directory
        os.makedirs(self.config.visualization_dir, exist_ok=True)
        
        # Extract feature columns
        feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
        
        # Check if we need dimensionality reduction before UMAP
        features = features_df[feature_cols].values
        feature_dim = features.shape[1]
        
        # Add normalized feature statistics
        self.logger.info(f"Feature statistics before dimensionality reduction:")
        self.logger.info(f"  Dimensions: {feature_dim}")
        self.logger.info(f"  Mean: {features.mean():.4f}")
        self.logger.info(f"  Std: {features.std():.4f}")
        self.logger.info(f"  Min: {features.min():.4f}")
        self.logger.info(f"  Max: {features.max():.4f}")
        
        # Setup labels and colors
        labels = features_df['bbox_name'].values if 'bbox_name' in features_df.columns else None
        bbox_to_color = None
        if labels is not None:
            # Create color mapping
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            bbox_to_color = {label: color for label, color in zip(unique_labels, colors)}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA if needed for high-dimensional data
        if use_pca and features.shape[1] > n_components:
            self.logger.info(f"Applying PCA to reduce dimensions from {features.shape[1]} to {n_components}")
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)
            self.logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
            
            # Plot PCA explained variance
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance')
            plt.grid(True)
            plt.savefig(os.path.join(self.config.visualization_dir, f"{save_prefix}_pca_variance.png"))
            plt.close()
        else:
            # Use standardized features directly
            features_pca = features_scaled
        
        # Apply UMAP
        self.logger.info("Applying UMAP for visualization...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        embedding = reducer.fit_transform(features_pca)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot points
        if labels is not None:
            # Plot with bbox colors
            for label in unique_labels:
                mask = labels == label
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    color=bbox_to_color[label],
                    label=label,
                    alpha=0.8,
                    s=50
                )
            plt.legend(title='BBox', loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
        else:
            # Plot without colors
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.8, s=50)
        
        plt.title(f"{save_prefix.capitalize()} Features - UMAP Projection")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.visualization_dir, f"{save_prefix}_umap.png"))
        plt.close()
        
        # Create 3D plot if requested
        if plot_3d:
            # Apply UMAP with 3 components
            reducer_3d = umap.UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            embedding_3d = reducer_3d.fit_transform(features_pca)
            
            # Create 3D figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            if labels is not None:
                # Plot with bbox colors
                for label in unique_labels:
                    mask = labels == label
                    ax.scatter(
                        embedding_3d[mask, 0],
                        embedding_3d[mask, 1],
                        embedding_3d[mask, 2],
                        color=bbox_to_color[label],
                        label=label,
                        alpha=0.8,
                        s=50
                    )
                ax.legend(title='BBox', loc='best')
            else:
                # Plot without colors
                ax.scatter(
                    embedding_3d[:, 0],
                    embedding_3d[:, 1],
                    embedding_3d[:, 2],
                    alpha=0.8,
                    s=50
                )
            
            ax.set_title(f"{save_prefix.capitalize()} Features - 3D UMAP Projection")
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.visualization_dir, f"{save_prefix}_umap_3d.png"))
            plt.close()
        
        # If base features provided, compare with contrastive features
        if base_features_df is not None:
            self.logger.info("Comparing base and contrastive features...")
            
            # Make sure both DataFrames have the same samples in the same order
            common_samples = set(features_df['synapse_id']).intersection(set(base_features_df['synapse_id']))
            self.logger.info(f"Found {len(common_samples)} common samples for comparison")
            
            # Filter to common samples
            features_df_common = features_df[features_df['synapse_id'].isin(common_samples)]
            base_features_df_common = base_features_df[base_features_df['synapse_id'].isin(common_samples)]
            
            # Sort by synapse_id for consistency
            features_df_common = features_df_common.sort_values('synapse_id')
            base_features_df_common = base_features_df_common.sort_values('synapse_id')
            
            # Extract feature columns
            contrastive_feature_cols = [col for col in features_df_common.columns if col.startswith('feature_')]
            base_feature_cols = [col for col in base_features_df_common.columns if col.startswith('feature_')]
            
            # Standardize both feature sets
            contrastive_features = features_df_common[contrastive_feature_cols].values
            base_features = base_features_df_common[base_feature_cols].values
            
            scaler_contrastive = StandardScaler()
            scaler_base = StandardScaler()
            
            contrastive_features_scaled = scaler_contrastive.fit_transform(contrastive_features)
            base_features_scaled = scaler_base.fit_transform(base_features)
            
            # Apply PCA if needed
            if use_pca:
                # For contrastive features
                if contrastive_features.shape[1] > n_components:
                    pca_contrastive = PCA(n_components=n_components)
                    contrastive_features_pca = pca_contrastive.fit_transform(contrastive_features_scaled)
                    self.logger.info(f"Contrastive PCA explained variance: {pca_contrastive.explained_variance_ratio_.sum():.4f}")
                else:
                    contrastive_features_pca = contrastive_features_scaled
                
                # For base features
                if base_features.shape[1] > n_components:
                    pca_base = PCA(n_components=n_components)
                    base_features_pca = pca_base.fit_transform(base_features_scaled)
                    self.logger.info(f"Base PCA explained variance: {pca_base.explained_variance_ratio_.sum():.4f}")
                else:
                    base_features_pca = base_features_scaled
            else:
                contrastive_features_pca = contrastive_features_scaled
                base_features_pca = base_features_scaled
            
            # Calculate feature similarity
            if contrastive_features_pca.shape[1] == base_features_pca.shape[1]:
                cosine_sims = np.diag(cosine_similarity(contrastive_features_pca, base_features_pca))
                self.logger.info(f"Average cosine similarity: {cosine_sims.mean():.4f}")
                
                # Plot cosine similarity distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(cosine_sims, kde=True)
                plt.title('Cosine Similarity Between Base and Contrastive Features')
                plt.xlabel('Cosine Similarity')
                plt.ylabel('Count')
                plt.grid(True)
                plt.savefig(os.path.join(self.config.visualization_dir, "feature_similarity.png"))
                plt.close()
            
            # Apply UMAP to combined features
            combined_features = np.vstack([base_features_pca, contrastive_features_pca])
            self.logger.info(f"Applying UMAP to combined features: {combined_features.shape}")
            
            # Create combined labels
            combined_labels = np.concatenate([
                np.array(['Base Model'] * len(base_features_pca)),
                np.array(['Contrastive Model'] * len(contrastive_features_pca))
            ])
            
            # Apply UMAP
            reducer_combined = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            embedding_combined = reducer_combined.fit_transform(combined_features)
            
            # Create figure for combined visualization
            plt.figure(figsize=(12, 10))
            
            # Get base and contrastive indices
            base_idx = combined_labels == 'Base Model'
            contrastive_idx = combined_labels == 'Contrastive Model'
            
            # Plot base features
            plt.scatter(
                embedding_combined[base_idx, 0],
                embedding_combined[base_idx, 1],
                color='blue',
                label='Base Model',
                alpha=0.7,
                s=50
            )
            
            # Plot contrastive features
            plt.scatter(
                embedding_combined[contrastive_idx, 0],
                embedding_combined[contrastive_idx, 1],
                color='red',
                label='Contrastive Model',
                alpha=0.7,
                s=50
            )
            
            plt.title("Base vs Contrastive Features - UMAP Projection")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.visualization_dir, "base_vs_contrastive.png"))
            plt.close()
            
            # Create a joint plot to show the separation
            if 'bbox_name' in features_df_common.columns:
                # Plot using seaborn's jointplot for each bbox
                bbox_names = features_df_common['bbox_name'].unique()
                
                for bbox in bbox_names:
                    # Get indices for this bbox
                    bbox_base_idx = (combined_labels == 'Base Model') & (np.array(base_features_df_common['bbox_name'].values) == bbox)
                    bbox_contrastive_idx = (combined_labels == 'Contrastive Model') & (np.array(features_df_common['bbox_name'].values) == bbox)
                    
                    if np.sum(bbox_base_idx) > 0 and np.sum(bbox_contrastive_idx) > 0:
                        plt.figure(figsize=(12, 10))
                        
                        # Plot base features
                        plt.scatter(
                            embedding_combined[bbox_base_idx, 0],
                            embedding_combined[bbox_base_idx, 1],
                            color='blue',
                            label=f'Base Model - {bbox}',
                            alpha=0.7,
                            s=50
                        )
                        
                        # Plot contrastive features
                        plt.scatter(
                            embedding_combined[bbox_contrastive_idx, 0],
                            embedding_combined[bbox_contrastive_idx, 1],
                            color='red',
                            label=f'Contrastive Model - {bbox}',
                            alpha=0.7,
                            s=50
                        )
                        
                        plt.title(f"Base vs Contrastive Features for {bbox} - UMAP Projection")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.config.visualization_dir, f"base_vs_contrastive_{bbox}.png"))
                        plt.close()
        
        self.logger.info("Feature visualization complete")
    
    def run_pipeline(self):
        """Run the complete contrastive learning pipeline."""
        self.logger.info("Starting contrastive learning pipeline...")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Extract features from base model for comparison
        base_checkpoint = self.config.base_checkpoint if hasattr(self.config, 'base_checkpoint') else None
        if base_checkpoint:
            self.logger.info(f"Extracting features from base model: {base_checkpoint}")
        self.setup_base_model()
            base_features_df = self.extract_base_features(base_checkpoint)
        else:
            self.logger.info("No base checkpoint specified, skipping base feature extraction")
            base_features_df = None
        
        # Step 3: Setup and train contrastive model
        self.setup_model()
            self.train_model()
        
        # Get last checkpoint path
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_path = os.path.join(checkpoint_dir, "contrastive_model_final.pt")
        
        # Step 4: Extract features using different methods and compare
        self.logger.info("Extracting features using different methods...")
        
        # Extract features using different layers/approaches
        feature_extraction_methods = [
            # (method_name, use_backbone, layer_idx)
            ("contrastive_projection", False, None),  # Use projection head (default)
            ("backbone_last", True, None),   # Use last backbone layer
            ("backbone_middle", True, 3)     # Use middle layer of backbone
        ]
        
        feature_dfs = {}
        
        for method_name, use_backbone, layer_idx in feature_extraction_methods:
            self.logger.info(f"Extracting features with method: {method_name}")
            features_df = self.extract_features(
                checkpoint_path, 
                use_backbone_features=use_backbone,
                layer_idx=layer_idx
            )
            
            # Save with method name
            output_path = os.path.join(self.config.results_dir, f"features_{method_name}.csv")
            features_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {method_name} features to: {output_path}")
            
            # Store for comparison
            feature_dfs[method_name] = features_df
            
            # Visualize
            self.visualize_features(
                features_df, 
                base_features_df,
                save_prefix=method_name
            )
        
        # Compare all methods together
        if len(feature_dfs) > 1:
            self.logger.info("Comparing different feature extraction methods...")
            self.compare_feature_methods(feature_dfs, base_features_df)
        
        self.logger.info("Contrastive learning pipeline completed successfully.")
    
    def compare_feature_methods(self, feature_dfs, base_features_df=None):
        """
        Compare different feature extraction methods.
        
        Args:
            feature_dfs (dict): Dictionary of DataFrames from different methods
            base_features_df (pd.DataFrame, optional): Base model features
        """
        self.logger.info("Comparing feature extraction methods...")
        
        # Create output directory
        comparison_dir = os.path.join(self.config.visualization_dir, "method_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Create combined visualization of all methods
        method_names = list(feature_dfs.keys())
        sample_counts = {name: len(df) for name, df in feature_dfs.items()}
        
        # Add base model if available
        if base_features_df is not None:
            method_names.append("base_model")
            sample_counts["base_model"] = len(base_features_df)
        
        # Ensure all methods have the same samples
        common_synapse_ids = set(feature_dfs[method_names[0]]['synapse_id'])
        for name in method_names[1:]:
            if name == "base_model":
                common_synapse_ids = common_synapse_ids.intersection(set(base_features_df['synapse_id']))
        else:
                common_synapse_ids = common_synapse_ids.intersection(set(feature_dfs[name]['synapse_id']))
        
        self.logger.info(f"Found {len(common_synapse_ids)} common samples across all methods")
        
        # Filter DataFrames to common samples
        filtered_dfs = {}
        for name in method_names:
            if name == "base_model":
                filtered_dfs[name] = base_features_df[base_features_df['synapse_id'].isin(common_synapse_ids)]
            else:
                filtered_dfs[name] = feature_dfs[name][feature_dfs[name]['synapse_id'].isin(common_synapse_ids)]
        
        # Process each DataFrame to extract features and scale
        processed_features = {}
        n_components = 50  # For PCA
        
        for name, df in filtered_dfs.items():
        # Extract features
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            features = df[feature_cols].values
            
            # Standardize
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA if needed
            if features.shape[1] > n_components:
                pca = PCA(n_components=n_components)
                features_pca = pca.fit_transform(features_scaled)
                self.logger.info(f"{name} PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
            else:
                features_pca = features_scaled
            
            processed_features[name] = features_pca
        
        # Calculate pairwise similarities
        similarity_matrix = np.zeros((len(method_names), len(method_names)))
        
        for i, name1 in enumerate(method_names):
            for j, name2 in enumerate(method_names):
                # Skip if same method
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                # If different dimensions, can't compute cosine similarity directly
                if processed_features[name1].shape[1] != processed_features[name2].shape[1]:
                    similarity_matrix[i, j] = np.nan
                    continue
                
                # Compute diagonal of cosine similarity matrix
                cosine_sims = np.diag(cosine_similarity(processed_features[name1], processed_features[name2]))
                similarity_matrix[i, j] = cosine_sims.mean()
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=method_names,
            yticklabels=method_names
        )
        plt.title('Average Feature Similarity Between Methods')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "method_similarity.png"))
        plt.close()
        
        # Combine all features into one array for UMAP
        all_features = []
        all_labels = []
        
        for name, features in processed_features.items():
            all_features.append(features)
            all_labels.extend([name] * len(features))
        
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        embedding = reducer.fit_transform(all_features)
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Define colors
        method_colors = {
            'contrastive_projection': 'red',
            'backbone_last': 'blue',
            'backbone_middle': 'green',
            'base_model': 'purple'
        }
        
        # Plot each method
        start_idx = 0
        for name in method_names:
            count = len(processed_features[name])
            end_idx = start_idx + count
            
            # Get color, default to black if not in dictionary
            color = method_colors.get(name, 'black')
            
            plt.scatter(
                embedding[start_idx:end_idx, 0],
                embedding[start_idx:end_idx, 1],
                color=color,
                label=name,
                alpha=0.7,
                s=50
            )
            
            start_idx = end_idx
        
        plt.title("Comparison of All Feature Extraction Methods - UMAP Projection")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "all_methods_umap.png"))
        plt.close()
        
        # If bbox_name is available, plot per bbox
        if 'bbox_name' in filtered_dfs[method_names[0]].columns:
            bbox_names = filtered_dfs[method_names[0]]['bbox_name'].unique()
            
            for bbox in bbox_names:
                # Create per-bbox plot
                plt.figure(figsize=(14, 12))
                
                start_idx = 0
                for name in method_names:
                    # Get indices for this bbox
                    bbox_mask = filtered_dfs[name]['bbox_name'] == bbox
                    count = len(processed_features[name])
                    
                    # Create mask for this bbox in the embedding
                    method_indices = np.zeros(len(all_labels), dtype=bool)
                    method_indices[start_idx:start_idx+count] = True
                    
                    # Get bbox indices within this method's section
                    bbox_indices_in_df = np.where(bbox_mask)[0]
                    
                    # Map to global indices
                    global_indices = method_indices.copy()
                    for idx in bbox_indices_in_df:
                        # Adjust index to global position
                        global_indices[start_idx + idx] = True
                    
                    # Get color
                    color = method_colors.get(name, 'black')
                    
                    # Plot points for this bbox and method
                    plt.scatter(
                        embedding[global_indices, 0],
                        embedding[global_indices, 1],
                        color=color,
                        label=f"{name} - {bbox}",
                        alpha=0.7,
                        s=50
                    )
                    
                    start_idx += count
                
                plt.title(f"Feature Comparison for {bbox} - UMAP Projection")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_dir, f"all_methods_{bbox}_umap.png"))
                plt.close()
        
        self.logger.info("Method comparison complete")


def main():
    """Main entry point for the contrastive learning pipeline."""
    parser = argparse.ArgumentParser(description="Run contrastive learning pipeline for synapse analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--base_checkpoint", type=str, default=None, help="Path to base model checkpoint")
    parser.add_argument("--synapse_config", type=str, default=None, help="Path to synapse configuration file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "extract", "visualize"], 
                       help="Pipeline mode")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        config = ContrastiveConfig.from_json(args.config)
    else:
        config = ContrastiveConfig()
    
    # Set base checkpoint from command line if provided
    if args.base_checkpoint:
        config.base_checkpoint = args.base_checkpoint
    
    # Load synapse configuration if provided
    synapse_cfg = None
    if args.synapse_config:
        synapse_cfg = SynapseConfig.from_json(args.synapse_config)
    
    # Create and run pipeline
    pipeline = ContrastivePipeline(config, synapse_cfg)
    
    # Run based on mode
    if args.mode == "train":
        pipeline.load_and_prepare_data()
        pipeline.setup_model()
            pipeline.train_model()
    elif args.mode == "extract":
        pipeline.load_and_prepare_data()
        pipeline.setup_model()
        features_df = pipeline.extract_features()
        print(f"Extracted features: {features_df.shape}")
    elif args.mode == "visualize":
        pipeline.load_and_prepare_data()
        
        # Load extracted features if they exist
        features_path = os.path.join(config.results_dir, "contrastive_features.csv")
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            pipeline.visualize_features(features_df)
        else:
            print(f"Features file not found: {features_path}")
    else:
        # Run full pipeline
        pipeline.run_pipeline()


if __name__ == "__main__":
    main() 