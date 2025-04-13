class VGG3DContrastive(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super().__init__()
        self.backbone = base_model
        self.projection_head = ProjectionHead(
            input_dim=512,  # VGG's feature dimension
            hidden_dim=512,
            output_dim=projection_dim
        )
        
        # Initially freeze the backbone
        self._freeze_backbone()
        
        # Track training phase
        self.training_phase = 'warmup'  # 'warmup', 'gradual', 'full'
        
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def _get_backbone_layers(self):
        """Get list of backbone layers for gradual unfreezing."""
        layers = []
        for name, module in self.backbone.named_children():
            if isinstance(module, nn.Sequential):
                for subname, submodule in module.named_children():
                    if isinstance(submodule, (nn.Conv3d, nn.Linear)):
                        layers.append((f"{name}.{subname}", submodule))
            elif isinstance(module, (nn.Conv3d, nn.Linear)):
                layers.append((name, module))
        return layers
    
    def set_training_phase(self, phase):
        """
        Set the training phase and adjust parameter freezing accordingly.
        
        Args:
            phase (str): One of 'warmup', 'gradual', or 'full'
        """
        self.training_phase = phase
        
        if phase == 'warmup':
            self._freeze_backbone()
            for param in self.projection_head.parameters():
                param.requires_grad = True
                
        elif phase == 'gradual':
            # Unfreeze last few layers of backbone
            backbone_layers = self._get_backbone_layers()
            for name, layer in backbone_layers[-3:]:  # Unfreeze last 3 layers
                for param in layer.parameters():
                    param.requires_grad = True
                    
        elif phase == 'full':
            self._unfreeze_backbone()
            
    def get_trainable_params(self):
        """Get parameters that require gradients based on current phase."""
        if self.training_phase == 'warmup':
            return self.projection_head.parameters()
        elif self.training_phase == 'gradual':
            params = list(self.projection_head.parameters())
            backbone_layers = self._get_backbone_layers()
            for _, layer in backbone_layers[-3:]:
                params.extend(layer.parameters())
            return params
        else:  # full
            return self.parameters()
            
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Project features
        projections = self.projection_head(features)
        
        return projections, features
    
    def extract_features(self, x, layer=None):
        """
        Extract features from the model.
        
        Args:
            x (torch.Tensor): Input tensor
            layer (str, optional): Specific layer to extract features from
            
        Returns:
            torch.Tensor: Extracted features
        """
        if layer is None:
            # Get final features
            features = self.backbone(x)
        else:
            # Get intermediate features
            features = self.backbone.extract_features(x, layer)
            
        return features
    
    def compare_features(self, x, reference_model):
        """
        Compare features between this model and a reference model.
        
        Args:
            x (torch.Tensor): Input tensor
            reference_model (nn.Module): Reference model (e.g., original VGG)
            
        Returns:
            dict: Dictionary containing feature comparison metrics
        """
        # Extract features from both models
        current_features = self.extract_features(x)
        reference_features = reference_model.extract_features(x)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            current_features.view(current_features.size(0), -1),
            reference_features.view(reference_features.size(0), -1)
        ).mean()
        
        # Calculate L2 distance
        l2_distance = F.mse_loss(current_features, reference_features)
        
        # Calculate feature statistics
        current_stats = {
            'mean': current_features.mean().item(),
            'std': current_features.std().item(),
            'min': current_features.min().item(),
            'max': current_features.max().item()
        }
        
        reference_stats = {
            'mean': reference_features.mean().item(),
            'std': reference_features.std().item(),
            'min': reference_features.min().item(),
            'max': reference_features.max().item()
        }
        
        return {
            'cosine_similarity': similarity.item(),
            'l2_distance': l2_distance.item(),
            'current_stats': current_stats,
            'reference_stats': reference_stats
        } 