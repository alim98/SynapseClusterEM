import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch.utils.checkpoint import checkpoint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the existing VGG3D model
from synapse.models import Vgg3D, load_model_from_checkpoint


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning. 
    Maps the high-dimensional representation to a lower-dimensional space.
    """
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128, small_batch_mode='layer_norm'):
        """
        Initialize the projection head with an MLP.
        
        Args:
            input_dim (int): Input dimension (from the encoder/backbone)
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            small_batch_mode (str): How to handle small batches: 'layer_norm', 'batch_norm', or 'instance_norm'
        """
        super(ProjectionHead, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.small_batch_mode = small_batch_mode
        
        # Choose normalization based on small_batch_mode
        if small_batch_mode == 'layer_norm':
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif small_batch_mode == 'instance_norm':
            self.norm1 = nn.InstanceNorm1d(hidden_dim)
        else:  # 'batch_norm'
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Forward pass of the projection head.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.layer1(x)
        
        # Special handling for batch size of 1 when using BatchNorm
        if self.small_batch_mode == 'batch_norm' and x.size(0) == 1:
            # Switch to eval mode temporarily to handle batch size of 1
            training_mode = self.norm1.training
            self.norm1.eval()
            x = self.norm1(x)
            # Restore previous training mode
            self.norm1.training = training_mode
        else:
            x = self.norm1(x)
            
        x = self.relu(x)
        x = self.layer2(x)
        return x


class CheckpointSequential(nn.Sequential):
    """A Sequential module that supports gradient checkpointing."""
    
    def __init__(self, *args):
        super(CheckpointSequential, self).__init__(*args)
        self.use_checkpointing = False
    
    def forward(self, input):
        if not self.use_checkpointing:
            return super(CheckpointSequential, self).forward(input)
        
        def custom_forward(*inputs):
            return super(CheckpointSequential, self).forward(inputs[0])
        
        return checkpoint(custom_forward, input)


class VGG3DContrastive(nn.Module):
    """
    Contrastive learning model with VGG3D backbone and projection head.
    """
    def __init__(self, checkpoint_path=None, proj_dim=128, input_size=(80, 80, 80),
                 fmaps=24, output_classes=7, input_fmaps=1, use_pretrained=True, small_batch_mode='layer_norm'):
        """
        Initialize the VGG3D contrastive model.
        
        Args:
            checkpoint_path (str, optional): Path to the VGG3D checkpoint
            proj_dim (int): Projection dimension
            input_size (tuple): Input size
            fmaps (int): Number of feature maps
            output_classes (int): Number of output classes
            input_fmaps (int): Number of input feature maps
            use_pretrained (bool): Whether to load pretrained weights
            small_batch_mode (str): How to handle small batches in projection head
        """
        super(VGG3DContrastive, self).__init__()
        
        # Initialize the VGG3D backbone
        self.backbone = Vgg3D(
            input_size=input_size,
            fmaps=fmaps,
            output_classes=output_classes,
            input_fmaps=input_fmaps
        )
        
        # Load pretrained weights if specified
        if use_pretrained and checkpoint_path is not None:
            print(f"Loading pretrained weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.backbone.load_state_dict(checkpoint)
        
        # Compute input dimension to the projection head
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_fmaps, *input_size)
            features = self.backbone.features(dummy_input)
            flattened_dim = features.view(1, -1).shape[1]
        
        # Create the projection head
        self.projection = ProjectionHead(
            input_dim=flattened_dim,
            hidden_dim=2048,
            output_dim=proj_dim,
            small_batch_mode=small_batch_mode
        )
        
        # Replace Sequential modules with CheckpointSequential
        if hasattr(self.backbone, 'features'):
            for i, module in enumerate(self.backbone.features):
                if isinstance(module, nn.Sequential):
                    # Create a new CheckpointSequential with the same layers
                    checkpoint_module = CheckpointSequential(*list(module.children()))
                    # Replace the original module
                    self.backbone.features[i] = checkpoint_module
        
    def forward(self, x, return_features=False):
        """
        Forward pass of the contrastive model.
        
        Args:
            x (torch.Tensor): Input tensor
            return_features (bool): Whether to return features
            
        Returns:
            torch.Tensor or tuple: Output tensor or (features, projections)
        """
        # Debug prints for input
        print(f"VGG3DContrastive forward - Input shape: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Get features from backbone
        features = self.backbone.features(x)
        print(f"Backbone features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
        
        flat_features = features.view(features.size(0), -1)
        print(f"Flattened features shape: {flat_features.shape}, range: [{flat_features.min():.3f}, {flat_features.max():.3f}]")
        
        # Pass through projection head
        projections = self.projection(flat_features)
        print(f"Projections shape: {projections.shape}, range: [{projections.min():.3f}, {projections.max():.3f}]")
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        if return_features:
            return flat_features, projections
        else:
            return projections
    
    def extract_features(self, x, layer_num=None):
        """
        Extract features from a specific layer of the VGG3D backbone.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_num (int, optional): Layer number to extract features from
            
        Returns:
            torch.Tensor: Features from the specified layer
        """
        # Debug prints for input
        print(f"extract_features - Input shape: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
        
        # If layer_num is None, extract features from the entire backbone
        if layer_num is None:
            features = self.backbone.features(x)
            print(f"extract_features - Full backbone features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
            return features.view(features.size(0), -1)
        
        # Extract features up to the specified layer
        features = x
        for i, layer in enumerate(self.backbone.features):
            features = layer(features)
            if i == layer_num:
                print(f"extract_features - Layer {layer_num} features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
                return features
        
        print(f"extract_features - Final features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
        return features.view(features.size(0), -1)


def initialize_contrastive_model(config):
    """
    Initialize the contrastive model with the given configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        VGG3DContrastive: Initialized contrastive model
    """
    checkpoint_path = config.model_path
    
    # Get the small_batch_mode from config or use a default value
    small_batch_mode = getattr(config, 'small_batch_mode', 'layer_norm')
    
    # Create the contrastive model
    model = VGG3DContrastive(
        checkpoint_path=checkpoint_path,
        proj_dim=config.projection_dim,
        input_size=(config.subvol_size, config.subvol_size, config.num_frames),
        fmaps=24,
        output_classes=7,
        input_fmaps=1,
        use_pretrained=True,
        small_batch_mode=small_batch_mode
    )
    
    # Freeze backbone if specified
    if hasattr(config, 'freeze_backbone') and config.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # If stage_to_finetune is specified, freeze all other stages
    if hasattr(config, 'stage_to_finetune') and config.stage_to_finetune >= 0:
        # Use the VGG3DStageExtractor to identify stage boundaries
        from vgg3d_stage_extractor import VGG3DStageExtractor
        extractor = VGG3DStageExtractor(model.backbone)
        stage_info = extractor.get_stage_info()
        
        # Freeze all parameters initially
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the parameters of the specified stage
        if config.stage_to_finetune in stage_info:
            start_idx, end_idx = stage_info[config.stage_to_finetune]['range']
            for i in range(start_idx, end_idx + 1):
                for param in model.backbone.features[i].parameters():
                    param.requires_grad = True
    
    return model 