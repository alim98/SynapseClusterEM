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

        if len(downsample_factors) != len(fmap_inc):
            raise ValueError("fmap_inc needs to have same length as downsample factors")
        if len(n_convolutions) != len(fmap_inc):
            raise ValueError("n_convolutions needs to have the same length as downsample factors")
        if np.any(np.array(n_convolutions) < 1):
            raise ValueError("Each layer must have at least one convolution")

        current_fmaps = input_fmaps
        current_size = np.array(input_size)

        layers = []
        for i, (df, nc) in enumerate(zip(downsample_factors, n_convolutions)):
            layers += [
                nn.Conv3d(current_fmaps, fmaps, kernel_size=3, padding=1),
                nn.BatchNorm3d(fmaps),
                nn.ReLU(inplace=True)
            ]

            for _ in range(nc - 1):
                layers += [
                    nn.Conv3d(fmaps, fmaps, kernel_size=3, padding=1),
                    nn.BatchNorm3d(fmaps),
                    nn.ReLU(inplace=True)
                ]

            layers.append(nn.MaxPool3d(df))

            current_fmaps = fmaps
            fmaps *= fmap_inc[i]

            current_size = np.floor(current_size / np.array(df))

        self.features = nn.Sequential(*layers)

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
            return x
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model_from_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model