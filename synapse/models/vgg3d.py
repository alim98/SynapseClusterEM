import numpy as np
import torch
import torch.nn as nn

class Vgg3D(nn.Module):
    def __init__(
        self,
        input_size=(80, 80, 80),
        fmaps=24,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        fmap_inc=(2, 2, 2, 2),
        n_convolutions=(4, 2, 2, 2),
        output_classes=7,
        input_fmaps=1,
    ):
        super(Vgg3D, self).__init__()

        # Validate input parameters
        if len(downsample_factors) != len(fmap_inc):
            raise ValueError("fmap_inc needs to have same length as downsample factors")
        if len(n_convolutions) != len(fmap_inc):
            raise ValueError("n_convolutions needs to have the same length as downsample factors")
        if np.any(np.array(n_convolutions) < 1):
            raise ValueError("Each layer must have at least one convolution")

        current_fmaps = input_fmaps
        current_size = np.array(input_size)

        # Feature extraction layers
        layers = []
        for i, (df, nc) in enumerate(zip(downsample_factors, n_convolutions)):
            # Convolution block
            layers += [
                nn.Conv3d(current_fmaps, fmaps, kernel_size=3, padding=1),
                nn.BatchNorm3d(fmaps),
                nn.ReLU(inplace=True)
            ]

            # Additional convolutions
            for _ in range(nc - 1):
                layers += [
                    nn.Conv3d(fmaps, fmaps, kernel_size=3, padding=1),
                    nn.BatchNorm3d(fmaps),
                    nn.ReLU(inplace=True)
                ]

            # Downsampling
            layers.append(nn.MaxPool3d(df))

            # Update feature map size
            current_fmaps = fmaps
            fmaps *= fmap_inc[i]

            # Update spatial dimensions
            current_size = np.floor(current_size / np.array(df))
            # logger.info(f"Block {i+1}: features {current_fmaps}, size {current_size}")

        self.features = nn.Sequential(*layers)

        # Classifier (not used for feature extraction)
        self.classifier = nn.Sequential(
            nn.Linear(int(np.prod(current_size)) * current_fmaps, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_classes),
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        if return_features:
            return x  # Return raw features before flattening
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model_from_checkpoint(model, checkpoint_path):
    """
    Load a Vgg3D model from a checkpoint file
    
    Args:
        model: The Vgg3D model instance to load weights into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        The loaded model
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # If checkpoint contains full model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model 