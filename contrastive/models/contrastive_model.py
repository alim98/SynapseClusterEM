import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the existing VGG3D model
from synapse.models import Vgg3D, load_model_from_checkpoint


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning. 
    Maps the high-dimensional representation to a lower-dimensional space.
    """
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        """
        Initialize the projection head with an MLP.
        
        Args:
            input_dim (int): Input dimension (from the encoder/backbone)
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
        """
        super(ProjectionHead, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class VGG3DContrastive(nn.Module):
    """
    Contrastive learning model with VGG3D backbone and projection head.
    """
    def __init__(self, vgg_model=None, checkpoint_path=None, proj_dim=128, input_size=(80, 80, 80),
                 fmaps=24, output_classes=7, input_fmaps=1, use_pretrained=True):
        """
        Initialize the VGG3D contrastive model.
        
        Args:
            vgg_model (nn.Module, optional): Pre-instantiated VGG3D model
            checkpoint_path (str, optional): Path to the VGG3D checkpoint
            proj_dim (int): Projection dimension
            input_size (tuple): Input size
            fmaps (int): Number of feature maps
            output_classes (int): Number of output classes
            input_fmaps (int): Number of input feature maps
            use_pretrained (bool): Whether to load pretrained weights
        """
        super(VGG3DContrastive, self).__init__()
        
        # Initialize the VGG3D backbone
        if vgg_model is None:
            self.backbone = Vgg3D(input_size=input_size, fmaps=fmaps, 
                                 output_classes=output_classes, input_fmaps=input_fmaps)
            
            # Load pretrained weights if specified
            if use_pretrained and checkpoint_path is not None:
                print(f"Loading pretrained weights from {checkpoint_path}")
                self.backbone = load_model_from_checkpoint(self.backbone, checkpoint_path)
        else:
            self.backbone = vgg_model
        
        # Compute input dimension to the projection head
        # This depends on the specific architecture of VGG3D
        # Assuming we'll flatten the features from the last convolutional layer
        # Get the output shape from the features part
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_fmaps, *input_size)
            features = self.backbone.features(dummy_input)
            flattened_dim = features.view(1, -1).shape[1]
        
        # Create the projection head
        self.projection = ProjectionHead(
            input_dim=flattened_dim,
            hidden_dim=2048,
            output_dim=proj_dim
        )
        
    def forward(self, x, return_features=False):
        """
        Forward pass of the contrastive model.
        
        Args:
            x (torch.Tensor): Input tensor
            return_features (bool): Whether to return features
            
        Returns:
            torch.Tensor or tuple: Output tensor or (features, projections)
        """
        # Get features from backbone
        features = self.backbone.features(x)
        flat_features = features.view(features.size(0), -1)
        
        # Pass through projection head
        projections = self.projection(flat_features)
        
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
        # If layer_num is None, extract features from the entire backbone
        if layer_num is None:
            features = self.backbone.features(x)
            return features.view(features.size(0), -1)
        
        # Extract features up to the specified layer
        features = x
        for i, layer in enumerate(self.backbone.features):
            features = layer(features)
            if i == layer_num:
                return features
        
        return features.view(features.size(0), -1)


def initialize_contrastive_model(config):
    """
    Initialize the contrastive model with the given configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        VGG3DContrastive: Initialized contrastive model
    """
    checkpoint_path = 'hemibrain_production.checkpoint'
    
    # Create the contrastive model
    model = VGG3DContrastive(
        checkpoint_path=checkpoint_path,
        proj_dim=config.proj_dim,
        input_size=(config.subvol_size, config.subvol_size, config.num_frames),
        fmaps=24,
        output_classes=7,
        input_fmaps=1,
        use_pretrained=True
    )
    
    # Freeze backbone if specified
    if config.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # If stage_to_finetune is specified, freeze all other stages
    if config.stage_to_finetune >= 0:
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