import torch
import torch.nn as nn
from synapse.models import Vgg3D, load_model_from_checkpoint

class VGG3DStageExtractor:
    """
    A high-level interface for extracting features from specific stages of the VGG3D model.
    
    This class allows extracting features from different stages of the VGG3D model
    without having to implement any methods. Users should implement the necessary
    functionality in their own code.
    
    Usage example:
    -------------
    extractor = VGG3DStageExtractor(model)
    stage1_features = extractor.extract_stage(1, input_tensor)
    stage2_features = extractor.extract_stage(2, input_tensor)
    """
    
    def __init__(self, model):
        """
        Initialize the VGG3DStageExtractor with a VGG3D model.
        
        Args:
            model: A VGG3D model instance
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        
        # Define stage boundaries - indices in the features Sequential module
        # Each stage ends with a MaxPool3D operation
        # This is based on the typical VGG architecture with 4 stages
        self.stage_boundaries = self._identify_stage_boundaries()
        
    def _identify_stage_boundaries(self):
        """
        Identify the boundaries of each stage in the VGG3D model.
        A stage is defined as ending with a MaxPool3D operation.
        
        Returns:
            dict: A dictionary mapping stage numbers to (start_idx, end_idx) tuples
        """
        boundaries = {}
        stage_num = 1
        start_idx = 0
        
        # Look for MaxPool3D operations to identify stage boundaries
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, nn.MaxPool3d):
                boundaries[stage_num] = (start_idx, i)
                stage_num += 1
                start_idx = i + 1
        
        # Handle the case where the last stage doesn't end with MaxPool3D
        if start_idx < len(self.model.features):
            boundaries[stage_num] = (start_idx, len(self.model.features) - 1)
        
        return boundaries
    
    def extract_stage(self, stage_number, inputs):
        """
        Extract features from a specific stage of the VGG3D model.
        
        Args:
            stage_number (int): The stage number to extract features from (1-based indexing)
            inputs (torch.Tensor): The input tensor to the model
            
        Returns:
            torch.Tensor: Features extracted from the specified stage
        """
        if stage_number not in self.stage_boundaries:
            raise ValueError(f"Stage {stage_number} not found. Available stages: {list(self.stage_boundaries.keys())}")
        
        start_idx, end_idx = self.stage_boundaries[stage_number]
        
        with torch.no_grad():
            x = inputs
            
            # Process all layers up to the end of the requested stage
            for i in range(end_idx + 1):
                x = self.model.features[i](x)
                
            return x
    
    def extract_layer(self, layer_number, inputs):
        """
        Extract features from a specific layer of the VGG3D model.
        
        Args:
            layer_number (int): The layer number to extract features from (0-based indexing)
            inputs (torch.Tensor): The input tensor to the model
            
        Returns:
            torch.Tensor: Features extracted from the specified layer
        """
        if layer_number < 0 or layer_number >= len(self.model.features):
            raise ValueError(f"Layer {layer_number} out of range. Model has {len(self.model.features)} layers.")
        
        with torch.no_grad():
            x = inputs
            
            # Process all layers up to the requested layer
            for i in range(layer_number + 1):
                x = self.model.features[i](x)
                
            return x
    
    def extract_layer_20(self, inputs):
        """
        Extract features specifically from layer 20, which has been identified
        as having the most attention on important areas.
        
        Args:
            inputs (torch.Tensor): The input tensor to the model
            
        Returns:
            torch.Tensor: Features extracted from layer 20
        """
        return self.extract_layer(20, inputs)
    
    def get_all_stages(self, inputs):
        """
        Extract features from all stages of the VGG3D model.
        
        Args:
            inputs (torch.Tensor): The input tensor to the model
            
        Returns:
            dict: A dictionary mapping stage numbers to feature tensors
        """
        results = {}
        
        with torch.no_grad():
            x = inputs
            
            for stage_num, (start_idx, end_idx) in self.stage_boundaries.items():
                # Process this stage
                stage_input = x.clone() if stage_num > 1 else x
                
                # If not the first stage, we need to process all previous layers
                if stage_num > 1:
                    prev_end = self.stage_boundaries[stage_num - 1][1]
                    for i in range(prev_end + 1, start_idx):
                        stage_input = self.model.features[i](stage_input)
                
                # Process the current stage
                for i in range(start_idx, end_idx + 1):
                    stage_input = self.model.features[i](stage_input)
                
                results[stage_num] = stage_input
                
                # Continue processing for the next stage
                if stage_num == 1:
                    for i in range(start_idx, end_idx + 1):
                        x = self.model.features[i](x)
            
        return results
    
    def get_stage_info(self):
        """
        Get information about the stages in the VGG3D model.
        
        Returns:
            dict: A dictionary containing information about each stage
        """
        info = {}
        
        for stage_num, (start_idx, end_idx) in self.stage_boundaries.items():
            layers = []
            for i in range(start_idx, end_idx + 1):
                layer = self.model.features[i]
                layer_type = type(layer).__name__
                
                if isinstance(layer, nn.Conv3d):
                    layer_info = f"{layer_type}(in={layer.in_channels}, out={layer.out_channels}, k={layer.kernel_size})"
                elif isinstance(layer, nn.MaxPool3d):
                    layer_info = f"{layer_type}(k={layer.kernel_size})"
                else:
                    layer_info = layer_type
                
                layers.append((i, layer_info))
            
            info[stage_num] = {
                'range': (start_idx, end_idx),
                'layers': layers
            }
        
        return info 
