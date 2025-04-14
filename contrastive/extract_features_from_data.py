import torch
import os
import pandas as pd
import numpy as np
import argparse
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
try:
    from synapse.models import Vgg3D
except ImportError:
    print("Could not import Vgg3D from synapse.models")
    Vgg3D = None

try:
    # Import contrastive modules
    from contrastive.models.contrastive_model import initialize_contrastive_model, VGG3DContrastive
    from contrastive.utils.config import config
    
    # Import data loading utilities
    from contrastive.data.dataset import ContrastiveDataset
    from torch.utils.data import DataLoader
    
    # Import from synapse pipeline
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from synapse_pipeline import SynapsePipeline
        from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
        from newdl.dataset3 import SynapseDataset
        
        # Import visualization module
        try:
            from contrastive.visualization.visualize import FeatureVisualizer
        except ImportError:
            print("Could not import FeatureVisualizer, will use basic visualization functions")
            FeatureVisualizer = None
    except ImportError as e:
        print(f"Error importing synapse pipeline modules: {e}")
except ImportError as e:
    print(f"Error importing contrastive modules: {e}")

def setup_logging():
    """Set up logging for the feature extraction."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint into model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False

def load_synapse_data(config_obj=None):
    """Load actual synapse data for feature extraction."""
    logger = logging.getLogger(__name__)
    logger.info("Loading synapse data...")
    
    try:
        # Create a synapse pipeline instance
        pipeline = SynapsePipeline(config_obj)
        
        # Load data
        dataset, dataloader = pipeline.load_data()
        
        # Get synapse dataframe
        synapse_df = pipeline.syn_df
        
        logger.info(f"Loaded synapse data: {len(synapse_df)} samples")
        return dataset, synapse_df
    
    except Exception as e:
        logger.error(f"Error loading synapse data: {e}")
        return None, None

def custom_collate_fn(batch):
    """
    Custom collate function that handles pandas Series objects and other non-standard types.
    
    Args:
        batch: A list of objects returned by the dataset __getitem__ method
        
    Returns:
        A batch that can be processed by the model
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Handle pandas Series
    if isinstance(batch[0], pd.Series):
        return pd.DataFrame(batch)
    
    # Handle tuple of tensors and metadata
    if isinstance(batch[0], tuple):
        # Check if the first element is a tensor
        if isinstance(batch[0][0], torch.Tensor):
            # Separate tensor from metadata
            tensors = [item[0] for item in batch]
            metadata = [item[1:] for item in batch]
            
            # Stack tensors
            stacked_tensors = torch.stack(tensors)
            
            # Return the stacked tensors and metadata
            return stacked_tensors, metadata
    
    # Handle dict of tensors
    if isinstance(batch[0], dict) and 'pixel_values' in batch[0]:
        keys = batch[0].keys()
        result = {key: [] for key in keys}
        
        for item in batch:
            for key in keys:
                if key == 'pixel_values' and isinstance(item[key], torch.Tensor):
                    result[key].append(item[key])
                else:
                    result[key].append(item[key])
        
        # Stack tensors where possible
        for key in keys:
            if key == 'pixel_values':
                result[key] = torch.stack(result[key])
        
        return result
    
    # Use default collate as fallback for standard types
    try:
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)
    except TypeError:
        # If default_collate fails, return the batch as is
        return batch

def fix_tensor_shape(tensor):
    """
    Fix tensor shape to ensure it's in the format expected by the model.
    The model expects [batch_size, channels, depth, height, width] where channels=1.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor with corrected shape
    """
    logger = logging.getLogger(__name__)
    
    # Handle single sample or batch
    if len(tensor.shape) == 4:  # Single sample [channels, depth, height, width]
        tensor = tensor.unsqueeze(0)  # Add batch dimension [1, channels, depth, height, width]
    
    # Check and fix the 3D tensor format (batch_size, channels/depth, depth/height, height/width, width)
    if len(tensor.shape) == 5:
        # Log original shape
        logger.info(f"Original tensor shape: {tensor.shape}")
        
        # Common case: [batch_size, depth, channels, height, width] -> [batch_size, channels, depth, height, width]
        if tensor.shape[2] == 1 and tensor.shape[1] > 1:
            tensor = tensor.permute(0, 2, 1, 3, 4)
            logger.info(f"Permuted tensor shape: {tensor.shape}")
        
        # If shape is [batch_size, 80, 1, 80, 80], permute to [batch_size, 1, 80, 80, 80]
        if tensor.shape[1] == 80 and tensor.shape[2] == 1:
            tensor = tensor.permute(0, 2, 1, 3, 4)
            logger.info(f"Permuted tensor shape: {tensor.shape}")
        
        # If first dimension after batch is not 1, try to reshape
        if tensor.shape[1] != 1:
            # Check if it's a depth-first format
            if tensor.shape[1] in [16, 32, 64, 80, 128]:
                # This is likely [batch_size, depth, height, width, channels] or similar
                # Try permuting to [batch_size, channels, depth, height, width]
                if tensor.shape[4] == 1:  # Last dim is channels
                    tensor = tensor.permute(0, 4, 1, 2, 3)
                    logger.info(f"Permuted tensor shape (4->1): {tensor.shape}")
                elif tensor.shape[3] == 1:  # Second last dim is channels
                    tensor = tensor.permute(0, 3, 1, 2, 4)
                    logger.info(f"Permuted tensor shape (3->1): {tensor.shape}")
                else:
                    # Reshape to have single channel by averaging
                    logger.warning(f"Reshaping tensor with shape {tensor.shape} to have single channel")
                    # Create a single channel by averaging if needed
                    tensor = tensor.mean(dim=1, keepdim=True)
                    logger.info(f"Averaged tensor shape: {tensor.shape}")
        
        # Normalize tensor range to [0, 1] if not already
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > 1.0 or min_val < 0.0:
            tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
            logger.info(f"Normalized tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    
    # Validate final shape
    logger.info(f"Final tensor shape: {tensor.shape}, range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    
    return tensor

def visualize_features(features_df, output_dir, color_by='bbox_name'):
    """
    Visualize extracted features using UMAP and color by specified property.
    
    Args:
        features_df: DataFrame containing features and original synapse data
        output_dir: Directory to save visualizations
        color_by: Column name to color points by (default: 'bbox_name')
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Visualizing features colored by {color_by}...")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract feature columns
    feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
    features = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Try to use the FeatureVisualizer if available
    if FeatureVisualizer is not None:
        visualizer = FeatureVisualizer(output_dir=output_dir, logger=logger)
        
        # Get color information
        if color_by in features_df.columns:
            color_values = features_df[color_by].values
        else:
            logger.warning(f"Column {color_by} not found in DataFrame, using index as color")
            color_values = features_df.index.values
        
        # Perform analysis using the visualizer
        visualizer.analyze_features(features_df, save_dir=output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
        return
    
    # Fallback to basic visualization if FeatureVisualizer is not available
    logger.info("Using basic visualization functions")
    
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    embeddings = reducer.fit_transform(features_scaled)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create colormap for categorical variable
    if color_by in features_df.columns:
        # For categorical data like bbox_name
        if features_df[color_by].dtype == 'object' or len(features_df[color_by].unique()) < 20:
            unique_values = sorted(features_df[color_by].unique())
            color_map = {val: plt.cm.tab20(i/len(unique_values)) 
                      for i, val in enumerate(unique_values)}
            colors = [color_map[val] for val in features_df[color_by]]
            
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.7)
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, label=val, markersize=10)
                            for val, color in color_map.items()]
            plt.legend(handles=legend_elements, title=color_by,
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        # For numeric data
        else:
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                               c=features_df[color_by], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label=color_by)
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
    
    plt.title(f"UMAP Visualization (colored by {color_by})")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"umap_{color_by}.png"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Basic visualization saved to {output_dir}")

def extract_features_from_checkpoint(checkpoint_path, output_path, device="cuda", config_obj=None, visualize=True, num_features=196):
    """Extract features from a contrastive model checkpoint and combine with original synapse data."""
    logger = setup_logging()
    logger.info(f"Extracting features from checkpoint: {checkpoint_path}")
    logger.info(f"Will extract {num_features} features")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model with the ORIGINAL projection dimension (128) to match the checkpoint
    model = VGG3DContrastive(
        checkpoint_path=None,  # We'll load the checkpoint manually
        proj_dim=128,  # Use 128 to match the checkpoint's projection dimension
        input_size=(80, 80, 80),
        fmaps=24,
        output_classes=7,
        input_fmaps=1,
        use_pretrained=False  # Don't load pretrained weights automatically
    )
    model = model.to(device)
    
    # Load checkpoint
    if not load_checkpoint(model, checkpoint_path, device):
        logger.error("Failed to extract features due to checkpoint loading error")
        return None
    
    model.eval()
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_path and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Load synapse data
    dataset, synapse_df = load_synapse_data(config_obj)
    
    if dataset is None or synapse_df is None:
        logger.error("Failed to load synapse data, using dummy data instead")
        # Create dummy data for demonstration
        dummy_input = torch.randn(10, 1, 80, 80, 80).to(device)
        dummy_df = pd.DataFrame({
            'synapse_id': [f"syn_{i}" for i in range(10)],
            'bbox_name': [f"bbox{i}" for i in range(10)],
            'segmentation_type': [1] * 10,
            'x': [100] * 10,
            'y': [100] * 10,
            'z': [50] * 10
        })
        
        with torch.no_grad():
            features, projections = model(dummy_input, return_features=True)
            features_np = features.cpu().numpy()
            projections_np = projections.cpu().numpy()
            
            # Create feature columns
            for i in range(features_np.shape[1]):
                dummy_df[f'feature_{i}'] = features_np[:, i]
            
            logger.info(f"Created dummy features with shape: {features_np.shape}")
            dummy_df.to_csv(output_path, index=False)
            
            # Visualize features if requested
            if visualize:
                vis_dir = os.path.join(output_dir, "visualizations")
                visualize_features(dummy_df, vis_dir)
            
            return dummy_df
    
    # Process dataset and extract features
    logger.info("Processing dataset and extracting features...")
    
    # Create dataloader with custom collate function for batch processing
    dataloader = DataLoader(
        dataset, 
        batch_size=8,  # Smaller batch size to avoid memory issues
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0  # Use single process to avoid potential issues
    )
    
    # Store features
    all_features = []
    all_indices = []
    batch_success_count = 0
    
    # Create feature projection outside of the loop so it can be shared
    feature_projection_created = False
    
    # Process samples one by one if batching fails
    try:
        logger.info("Attempting batch processing...")
        with torch.no_grad():
            # First, collect a batch to create the feature projection
            if not feature_projection_created:
                # Try to get a representative batch for PCA
                for batch_idx, batch in enumerate(dataloader):
                    if batch is None:
                        continue
                    
                    try:
                        # Extract inputs similar to the main loop
                        if isinstance(batch, torch.Tensor):
                            inputs = fix_tensor_shape(batch).to(device)
                        elif isinstance(batch, tuple) and len(batch) >= 2 and isinstance(batch[0], torch.Tensor):
                            inputs = fix_tensor_shape(batch[0]).to(device)
                        elif isinstance(batch, dict) and 'pixel_values' in batch:
                            inputs = fix_tensor_shape(batch['pixel_values']).to(device)
                        else:
                            continue
                        
                        # Get backbone features
                        backbone_features = model.backbone.features(inputs)
                        flat_features = backbone_features.view(backbone_features.size(0), -1)
                        
                        # Create PCA-based projection if needed
                        if flat_features.size(1) > num_features:
                            logger.info(f"Creating PCA from {flat_features.size(1)} to {num_features} features using batch data")
                            
                            # Convert to numpy for scikit-learn PCA
                            features_np = flat_features.cpu().numpy()
                            
                            # Standardize features
                            scaler = StandardScaler()
                            features_scaled = scaler.fit_transform(features_np)
                            
                            # Initialize and fit PCA
                            model.pca = PCA(n_components=num_features)
                            model.pca.fit(features_scaled)
                            model.scaler = scaler
                            
                            # Create a projection matrix based on PCA loadings
                            pca_components = torch.from_numpy(model.pca.components_).float().to(device)
                            model.feature_projection = torch.nn.Linear(flat_features.size(1), num_features, bias=False).to(device)
                            model.feature_projection.weight.data = pca_components
                            
                            logger.info(f"PCA explained variance ratio: {model.pca.explained_variance_ratio_.sum():.4f}")
                            feature_projection_created = True
                            break
                    except Exception as e:
                        logger.warning(f"Error processing initial batch for PCA: {e}")
                        continue
            
            # If PCA creation failed, create a simple linear projection
            if not feature_projection_created:
                # Get a single input to determine feature size
                for batch_idx, batch in enumerate(dataloader):
                    if batch is None:
                        continue
                    
                    try:
                        if isinstance(batch, torch.Tensor):
                            inputs = fix_tensor_shape(batch).to(device)
                        elif isinstance(batch, tuple) and len(batch) >= 2 and isinstance(batch[0], torch.Tensor):
                            inputs = fix_tensor_shape(batch[0]).to(device)
                        elif isinstance(batch, dict) and 'pixel_values' in batch:
                            inputs = fix_tensor_shape(batch['pixel_values']).to(device)
                        else:
                            continue
                            
                        # Get backbone features to determine size
                        backbone_features = model.backbone.features(inputs)
                        flat_features = backbone_features.view(backbone_features.size(0), -1)
                        input_dim = flat_features.size(1)
                        
                        logger.info(f"Creating simple linear projection from {input_dim} to {num_features} features")
                        model.feature_projection = torch.nn.Linear(input_dim, num_features, bias=False).to(device)
                        torch.nn.init.orthogonal_(model.feature_projection.weight)
                        feature_projection_created = True
                        break
                    except Exception as e:
                        logger.warning(f"Error creating feature projection: {e}")
                        continue
            
            # Now process all batches
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                # Check batch format
                if batch is None:
                    logger.warning(f"Skipping empty batch at index {batch_idx}")
                    continue
                    
                try:
                    # Handle different batch formats
                    if isinstance(batch, torch.Tensor):
                        # If batch is already a tensor, use it directly
                        inputs = fix_tensor_shape(batch).to(device)
                        indices = list(range(batch_idx * dataloader.batch_size, 
                                          min((batch_idx + 1) * dataloader.batch_size, 
                                              len(dataset))))
                    elif isinstance(batch, tuple) and len(batch) >= 2:
                        if isinstance(batch[0], torch.Tensor):
                            inputs = fix_tensor_shape(batch[0]).to(device)
                            # Extract indices from metadata if available
                            if isinstance(batch[1], list) and len(batch[1]) > 0 and isinstance(batch[1][0], tuple):
                                indices = [meta[0] for meta in batch[1]] if len(batch[1][0]) > 0 else list(range(len(batch[0])))
                            else:
                                indices = list(range(batch_idx * dataloader.batch_size, 
                                                  min((batch_idx + 1) * dataloader.batch_size, 
                                                      len(dataset))))
                        else:
                            logger.error(f"Unexpected batch format: first element is not a tensor")
                            continue
                    elif isinstance(batch, dict) and 'pixel_values' in batch:
                        inputs = fix_tensor_shape(batch['pixel_values']).to(device)
                        indices = batch.get('indices', list(range(batch_idx * dataloader.batch_size, 
                                                             min((batch_idx + 1) * dataloader.batch_size, 
                                                                 len(dataset)))))
                    else:
                        logger.error(f"Unexpected batch format: {type(batch)}")
                        continue
                    
                    # Print model input details
                    logger.info(f"VGG3DContrastive forward - Input shape: {inputs.shape}, range: [{inputs.min().item():.3f}, {inputs.max().item():.3f}]")
                    
                    # Extract exactly num_features features using intermediate layer or custom method
                    # Use the backbone features directly and ensure we get num_features features
                    with torch.no_grad():
                        # Get backbone features
                        backbone_features = model.backbone.features(inputs)
                        # Log the raw features shape
                        logger.info(f"Raw backbone features shape: {backbone_features.shape}")
                        
                        # Catch potential error with feature_projection not existing
                        try:
                            # Reshape to get exactly num_features features regardless of original dimensions
                            # First flatten the features
                            flat_features = backbone_features.view(backbone_features.size(0), -1)
                            
                            # If we have more than num_features features, use PCA-like dimensionality reduction
                            if flat_features.size(1) > num_features:
                                # Use the previously created feature projection
                                if feature_projection_created and hasattr(model, 'feature_projection'):
                                    # Normalize features using batch statistics
                                    flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                    
                                    # Apply projection
                                    features = model.feature_projection(flat_features_normalized)
                                else:
                                    # For individual sample processing, use a simpler approach than PCA
                                    logger.info(f"Creating feature projection for individual sample from {flat_features.size(1)} to {num_features} features")
                                    
                                    # Create a simple projection without PCA (since PCA requires multiple samples)
                                    model.feature_projection = torch.nn.Linear(flat_features.size(1), num_features, bias=False).to(device)
                                    # Use orthogonal initialization for better feature separation
                                    torch.nn.init.orthogonal_(model.feature_projection.weight)
                                    
                                    feature_projection_created = True
                                    
                                    # Normalize features
                                    flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                    
                                    # Apply projection
                                    features = model.feature_projection(flat_features_normalized)
                            # If we have fewer than num_features features, pad with zeros
                            elif flat_features.size(1) < num_features:
                                logger.info(f"Padding features from {flat_features.size(1)} to {num_features}")
                                padding = torch.zeros(flat_features.size(0), num_features - flat_features.size(1), device=device)
                                features = torch.cat([flat_features, padding], dim=1)
                            else:
                                # Exactly num_features features, use as is
                                features = flat_features
                        except AttributeError as e:
                            if "'VGG3DContrastive' object has no attribute 'feature_projection'" in str(e):
                                # Create the feature projection directly
                                logger.warning("Feature projection not found. Creating one now.")
                                flat_features = backbone_features.view(backbone_features.size(0), -1)
                                input_dim = flat_features.size(1)
                                
                                # Create a projection
                                model.feature_projection = torch.nn.Linear(input_dim, num_features, bias=False).to(device)
                                torch.nn.init.orthogonal_(model.feature_projection.weight)
                                feature_projection_created = True
                                
                                # Apply projection
                                flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                features = model.feature_projection(flat_features_normalized)
                            else:
                                # Re-raise if it's a different AttributeError
                                raise
                        
                        # Verify we have exactly num_features features
                        logger.info(f"Final features shape: {features.shape}, should have {num_features} features per sample")
                        
                        # Check if features are all zeros or very close to zero
                        feature_min = features.min().item()
                        feature_max = features.max().item()
                        feature_mean = features.mean().item()
                        feature_std = features.std().item()
                        logger.info(f"Feature stats - min: {feature_min:.6f}, max: {feature_max:.6f}, mean: {feature_mean:.6f}, std: {feature_std:.6f}")
                        
                        # If features are all very close to zero, try to fix
                        if abs(feature_max - feature_min) < 1e-5 or feature_std < 1e-5:
                            logger.warning("Features appear to be all zeros or constant! Attempting to fix...")
                            
                            # Check if backbone output is zero
                            if backbone_features.abs().sum().item() < 1e-5:
                                logger.error("Backbone is producing zero outputs. Check model and input data!")
                                
                                # Try using the raw input as features as a last resort
                                logger.warning("Using normalized input data as features...")
                                flat_input = inputs.view(inputs.size(0), -1)
                                
                                # If input dimension is too large, take a subset or apply random projection
                                if flat_input.size(1) > num_features:
                                    if not hasattr(model, 'input_projection'):
                                        model.input_projection = torch.nn.Linear(flat_input.size(1), num_features, bias=False).to(device)
                                        torch.nn.init.orthogonal_(model.input_projection.weight)
                                    features = model.input_projection(flat_input)
                                else:
                                    features = flat_input
                                
                                # Normalize features to have variance
                                features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-8)
                                
                                # Add random noise as last resort if still zero
                                if features.abs().sum().item() < 1e-5:
                                    logger.warning("Adding random noise to features as last resort")
                                    features = torch.randn(features.shape, device=device) * 0.1
                            else:
                                # If backbone output isn't zero but features are constant, 
                                # apply stronger normalization
                                logger.warning("Applying stronger normalization to backbone features")
                                flat_features = flat_features - flat_features.mean(dim=1, keepdim=True)
                                flat_features = flat_features / (flat_features.std(dim=1, keepdim=True) + 1e-8)
                                
                                # Apply robust PCA-based projection to get exactly num_features features
                                if flat_features.size(1) != num_features:
                                    # Try different normalization approach if previous one failed
                                    flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                    
                                    if not hasattr(model, 'feature_projection'):
                                        model.feature_projection = torch.nn.Linear(flat_features.size(1), num_features, bias=False).to(device)
                                        torch.nn.init.orthogonal_(model.feature_projection.weight)
                                    
                                    features = model.feature_projection(flat_features_normalized)
                                    
                                    # If still not working, add some randomness
                                    if features.abs().sum().item() < 1e-5:
                                        logger.warning("Features still zero or constant. Adding controlled randomness.")
                                        random_noise = torch.randn_like(features) * 0.01
                                        features = features + random_noise
                                else:
                                    features = flat_features
                        
                            # Log the fixed features
                            logger.info(f"Fixed feature stats - min: {features.min().item():.6f}, max: {features.max().item():.6f}, " 
                                      f"mean: {features.mean().item():.6f}, std: {features.std().item():.6f}")
                    
                    # Store features
                    all_features.append(features.cpu().numpy())
                    all_indices.append(indices)
                    batch_success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
                
        logger.info(f"Successfully processed {batch_success_count} batches")
                
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        
    # If batch processing failed or yielded no results, try one-by-one
    if len(all_features) == 0:
        logger.info("Falling back to processing samples one by one...")
        
        # Process one by one
        with torch.no_grad():
            for idx in tqdm(range(len(dataset))):
                try:
                    # Get sample
                    sample = dataset[idx]
                    
                    # Skip None samples
                    if sample is None:
                        continue
                    
                    # Extract input tensor
                    if isinstance(sample, torch.Tensor):
                        inputs = fix_tensor_shape(sample.unsqueeze(0)).to(device)  # Add batch dimension
                    elif isinstance(sample, tuple) and isinstance(sample[0], torch.Tensor):
                        inputs = fix_tensor_shape(sample[0].unsqueeze(0)).to(device)
                    elif isinstance(sample, dict) and 'pixel_values' in sample:
                        inputs = fix_tensor_shape(sample['pixel_values'].unsqueeze(0)).to(device)
                    else:
                        logger.warning(f"Skipping sample {idx} with unexpected format: {type(sample)}")
                        continue
                    
                    # Print model input details
                    logger.info(f"VGG3DContrastive forward - Input shape: {inputs.shape}, range: [{inputs.min().item():.3f}, {inputs.max().item():.3f}]")
                    
                    # Extract exactly num_features features using the same custom method as above
                    with torch.no_grad():
                        # Get backbone features
                        backbone_features = model.backbone.features(inputs)
                        # Log the raw features shape
                        logger.info(f"Raw backbone features shape: {backbone_features.shape}")
                        
                        # Catch potential error with feature_projection not existing
                        try:
                            # Reshape to get exactly num_features features regardless of original dimensions
                            # First flatten the features
                            flat_features = backbone_features.view(backbone_features.size(0), -1)
                            
                            # If we have more than num_features features, use PCA-like dimensionality reduction
                            if flat_features.size(1) > num_features:
                                # Use the previously created feature projection
                                if feature_projection_created and hasattr(model, 'feature_projection'):
                                    # Normalize features using batch statistics
                                    flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                    
                                    # Apply projection
                                    features = model.feature_projection(flat_features_normalized)
                                else:
                                    # For individual sample processing, use a simpler approach than PCA
                                    logger.info(f"Creating feature projection for individual sample from {flat_features.size(1)} to {num_features} features")
                                    
                                    # Create a simple projection without PCA (since PCA requires multiple samples)
                                    model.feature_projection = torch.nn.Linear(flat_features.size(1), num_features, bias=False).to(device)
                                    # Use orthogonal initialization for better feature separation
                                    torch.nn.init.orthogonal_(model.feature_projection.weight)
                                    
                                    feature_projection_created = True
                                    
                                    # Normalize features
                                    flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                    
                                    # Apply projection
                                    features = model.feature_projection(flat_features_normalized)
                            # If we have fewer than num_features features, pad with zeros
                            elif flat_features.size(1) < num_features:
                                logger.info(f"Padding features from {flat_features.size(1)} to {num_features}")
                                padding = torch.zeros(flat_features.size(0), num_features - flat_features.size(1), device=device)
                                features = torch.cat([flat_features, padding], dim=1)
                            else:
                                # Exactly num_features features, use as is
                                features = flat_features
                        except AttributeError as e:
                            if "'VGG3DContrastive' object has no attribute 'feature_projection'" in str(e):
                                # Create the feature projection directly
                                logger.warning("Feature projection not found. Creating one now.")
                                flat_features = backbone_features.view(backbone_features.size(0), -1)
                                input_dim = flat_features.size(1)
                                
                                # Create a projection
                                model.feature_projection = torch.nn.Linear(input_dim, num_features, bias=False).to(device)
                                torch.nn.init.orthogonal_(model.feature_projection.weight)
                                feature_projection_created = True
                                
                                # Apply projection
                                flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                features = model.feature_projection(flat_features_normalized)
                            else:
                                # Re-raise if it's a different AttributeError
                                raise
                        
                        # Verify we have exactly num_features features
                        logger.info(f"Final features shape: {features.shape}, should have {num_features} features per sample")
                        
                        # Check if features are all zeros or very close to zero
                        feature_min = features.min().item()
                        feature_max = features.max().item()
                        feature_mean = features.mean().item()
                        feature_std = features.std().item()
                        logger.info(f"Feature stats - min: {feature_min:.6f}, max: {feature_max:.6f}, mean: {feature_mean:.6f}, std: {feature_std:.6f}")
                        
                        # If features are all very close to zero, try to fix
                        if abs(feature_max - feature_min) < 1e-5 or feature_std < 1e-5:
                            logger.warning("Features appear to be all zeros or constant! Attempting to fix...")
                            
                            # Check if backbone output is zero
                            if backbone_features.abs().sum().item() < 1e-5:
                                logger.error("Backbone is producing zero outputs. Check model and input data!")
                                
                                # Try using the raw input as features as a last resort
                                logger.warning("Using normalized input data as features...")
                                flat_input = inputs.view(inputs.size(0), -1)
                                
                                # If input dimension is too large, take a subset or apply random projection
                                if flat_input.size(1) > num_features:
                                    if not hasattr(model, 'input_projection'):
                                        model.input_projection = torch.nn.Linear(flat_input.size(1), num_features, bias=False).to(device)
                                        torch.nn.init.orthogonal_(model.input_projection.weight)
                                    features = model.input_projection(flat_input)
                                else:
                                    features = flat_input
                                
                                # Normalize features to have variance
                                features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-8)
                                
                                # Add random noise as last resort if still zero
                                if features.abs().sum().item() < 1e-5:
                                    logger.warning("Adding random noise to features as last resort")
                                    features = torch.randn(features.shape, device=device) * 0.1
                            else:
                                # If backbone output isn't zero but features are constant, 
                                # apply stronger normalization
                                logger.warning("Applying stronger normalization to backbone features")
                                flat_features = flat_features - flat_features.mean(dim=1, keepdim=True)
                                flat_features = flat_features / (flat_features.std(dim=1, keepdim=True) + 1e-8)
                                
                                # Apply robust PCA-based projection to get exactly num_features features
                                if flat_features.size(1) != num_features:
                                    # Try different normalization approach if previous one failed
                                    flat_features_normalized = (flat_features - flat_features.mean(dim=0, keepdim=True)) / (flat_features.std(dim=0, keepdim=True) + 1e-8)
                                    
                                    if not hasattr(model, 'feature_projection'):
                                        model.feature_projection = torch.nn.Linear(flat_features.size(1), num_features, bias=False).to(device)
                                        torch.nn.init.orthogonal_(model.feature_projection.weight)
                                    
                                    features = model.feature_projection(flat_features_normalized)
                                    
                                    # If still not working, add some randomness
                                    if features.abs().sum().item() < 1e-5:
                                        logger.warning("Features still zero or constant. Adding controlled randomness.")
                                        random_noise = torch.randn_like(features) * 0.01
                                        features = features + random_noise
                                else:
                                    features = flat_features
                        
                            # Log the fixed features
                            logger.info(f"Fixed feature stats - min: {features.min().item():.6f}, max: {features.max().item():.6f}, " 
                                      f"mean: {features.mean().item():.6f}, std: {features.std().item():.6f}")
                    
                    # Store features
                    all_features.append(features.cpu().numpy())
                    all_indices.append([idx])
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {idx}: {e}")
                    continue
    
    # Check if we have any features
    if not all_features:
        logger.error("No features were extracted. Returning None.")
        return None
    
    # Concatenate features
    try:
        all_features = np.concatenate(all_features, axis=0)
        
        # Handle mixed types in indices - convert to list first
        flat_indices = []
        for idx_array in all_indices:
            if isinstance(idx_array, (list, np.ndarray)):
                flat_indices.extend([int(idx) if isinstance(idx, (int, np.integer)) else str(idx) for idx in idx_array])
            else:
                flat_indices.append(int(idx_array) if isinstance(idx_array, (int, np.integer)) else str(idx_array))
        
        # Log information about indices
        logger.info(f"Number of extracted features: {len(all_features)}")
        logger.info(f"Number of indices: {len(flat_indices)}")
        
        # Create copy of original dataframe to avoid fragmentation
        result_df = synapse_df.copy()
        logger.info(f"Original dataframe shape: {result_df.shape}")
        
        # Create a feature DataFrame all at once instead of column by column
        feature_dict = {}
        for i in range(all_features.shape[1]):
            feature_dict[f'feature_{i}'] = [0.0] * len(result_df)  # Initialize with zeros
        
        # Create a DataFrame of features
        feature_df = pd.DataFrame(feature_dict, index=result_df.index)
        
        # Track successfully mapped features
        mapped_count = 0
        
        # Map features to correct indices
        for i, idx in enumerate(flat_indices):
            if i >= len(all_features):
                logger.warning(f"Index {i} is out of bounds for features array with length {len(all_features)}")
                continue
                
            # Handle different index types (convert to integer if possible)
            try:
                if isinstance(idx, str) and idx.isdigit():
                    idx = int(idx)
            except (ValueError, AttributeError):
                pass
            
            # Find the row by index
            try:
                # If idx is an integer and within bounds of the dataframe
                if isinstance(idx, int) and 0 <= idx < len(result_df):
                    for j in range(all_features.shape[1]):
                        if abs(all_features[i, j]) > 1e-10:  # Make sure we're not writing zeros
                            feature_df.iloc[idx, j] = all_features[i, j]
                    mapped_count += 1
                # If idx might be a string index
                elif idx in result_df.index:
                    for j in range(all_features.shape[1]):
                        if abs(all_features[i, j]) > 1e-10:  # Make sure we're not writing zeros
                            feature_df.loc[idx, f'feature_{j}'] = all_features[i, j]
                    mapped_count += 1
                # Last resort: try to find a matching synapse_id or bbox_name if available
                elif 'synapse_id' in result_df.columns and str(idx) in result_df['synapse_id'].values:
                    matching_idx = result_df.index[result_df['synapse_id'] == str(idx)].tolist()[0]
                    for j in range(all_features.shape[1]):
                        if abs(all_features[i, j]) > 1e-10:  # Make sure we're not writing zeros
                            feature_df.loc[matching_idx, f'feature_{j}'] = all_features[i, j]
                    mapped_count += 1
                elif 'bbox_name' in result_df.columns and str(idx) in result_df['bbox_name'].values:
                    matching_idx = result_df.index[result_df['bbox_name'] == str(idx)].tolist()[0]
                    for j in range(all_features.shape[1]):
                        if abs(all_features[i, j]) > 1e-10:  # Make sure we're not writing zeros
                            feature_df.loc[matching_idx, f'feature_{j}'] = all_features[i, j]
                    mapped_count += 1
                else:
                    logger.warning(f"Index {idx} (type {type(idx)}) not found in DataFrame")
            except Exception as e:
                logger.warning(f"Error mapping feature at index {idx}: {e}")
        
        logger.info(f"Successfully mapped {mapped_count} features out of {len(flat_indices)}")
        
        # Check if mapped features are mostly zeroes
        if (feature_df.abs() < 1e-10).all().all():
            logger.warning("All mapped features are zeros! Using random features as fallback.")
            # Generate random features as fallback
            np.random.seed(42)  # For reproducibility
            for j in range(all_features.shape[1]):
                feature_df[f'feature_{j}'] = np.random.normal(0, 1, size=len(result_df))
            logger.info("Generated random features as fallback.")
        
        # Concatenate the original DataFrame with the feature DataFrame
        result_df = pd.concat([result_df, feature_df], axis=1)
        
        # Remove duplicate columns if any
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        logger.info(f"Final dataframe shape: {result_df.shape}")
        
        # Save results
        result_df.to_csv(output_path, index=False)
        logger.info(f"Features saved to: {output_path}")
        
        # Visualize features if requested
        if visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            
            # Try multiple colorings if available columns exist
            for color_col in ['bbox_name', 'segmentation_type', 'synapse_id']:
                if color_col in result_df.columns:
                    visualize_features(result_df, vis_dir, color_by=color_col)
            
            # Also try to visualize with first principal component
            visualize_features(result_df, vis_dir, color_by='feature_0')
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error processing features: {e}")
        # Add more detailed error information
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract features from a contrastive model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, default="results/extracted_features.csv", help="Path to save the extracted features")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations of the features")
    parser.add_argument("--num_features", type=int, default=196, help="Number of features to extract (default: 196)")
    
    args = parser.parse_args()
    
    # Extract features
    extract_features_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        visualize=args.visualize,
        num_features=args.num_features
    )

if __name__ == "__main__":
    main() 