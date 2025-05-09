"""
Patch for inference.py to handle different tensor dimensions
"""

import torch
import numpy as np
from tqdm import tqdm

def patch_extract_stage_specific_features(model, dataset, config, layer_num=20, pooling_method='avg'):
    
    print(f"Using stage_specific feature extraction method with {pooling_method} pooling")
    print(f"Extracting features from layer {layer_num}")
    
    try:
        
        from vgg3d_stage_extractor import VGG3DStageExtractor
    except ImportError:
        
        print("Using vgg3d_stage_extractor_with_manual as fallback")
        from vgg3d_stage_extractor_with_manual import VGG3DStageExtractor
    
    extractor = VGG3DStageExtractor(model)
    
    
    stage_info = extractor.get_stage_info()
    print("VGG3D Stage Information:")
    for stage_num, info in stage_info.items():
        print(f"Stage {stage_num}: Layers {info['range'][0]}-{info['range'][1]}")
    
    
    layer_to_stage = {}
    for stage_num, info in stage_info.items():
        start_idx, end_idx = info['range']
        for i in range(start_idx, end_idx + 1):
            layer_to_stage[i] = stage_num
    
    
    if layer_num not in layer_to_stage:
        max_layer = max(layer_to_stage.keys())
        raise ValueError(f"Layer {layer_num} not found. Valid layer numbers are 0-{max_layer}")
    
    stage = layer_to_stage.get(layer_num)
    print(f"Layer {layer_num} is in Stage {stage}")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )
    
    
    features = []
    metadata = []
    
    
    if pooling_method == 'avg' or not pooling_method:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting layer {layer_num} features", unit="batch"):
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                
                
                print(f"Input tensor shape: {pixels.shape}")
                
                
                
                if len(pixels.shape) == 6:
                    
                    pixels = pixels.squeeze(1)
                    print(f"Squeezed tensor shape: {pixels.shape}")
                
                
                
                
                inputs = pixels.to(device)
                print(f"Input tensor to model shape: {inputs.shape}")
                
                
                batch_features = extractor.extract_layer(layer_num, inputs)
                print(f"Extracted features shape: {batch_features.shape}")
                
                
                batch_size = batch_features.shape[0]
                num_channels = batch_features.shape[1]
                
                
                batch_features_reshaped = batch_features.reshape(batch_size, num_channels, -1)
                
                
                pooled_features = torch.mean(batch_features_reshaped, dim=2)
                
                
                features_np = pooled_features.cpu().numpy()
                
                features.append(features_np)
                metadata.extend(zip(names, info))
        
        
        features = np.concatenate(features, axis=0)
        
        
        import pandas as pd
        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])
        
        
        feature_columns = [f'layer{layer_num}_feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        
        combined_df = pd.concat([metadata_df, features_df], axis=1)
        
        return combined_df
    else:
        
        print(f"Pooling method {pooling_method} not implemented in patch")
        return None

def patch_extract_features(model, dataset, config, pooling_method='avg'):
    """
    Patched version of extract_features that handles different tensor dimensions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )

    features = []
    metadata = []

    
    if pooling_method == 'avg' or not pooling_method:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", unit="batch"):
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                
                
                print(f"Input tensor shape: {pixels.shape}")
                
                
                
                if len(pixels.shape) == 6:
                    
                    pixels = pixels.squeeze(1)
                    print(f"Squeezed tensor shape: {pixels.shape}")
                
                
                
                
                inputs = pixels.to(device)
                print(f"Input tensor to model shape: {inputs.shape}")

                batch_features = model.features(inputs)
                pooled_features = torch.nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)

                batch_features_np = pooled_features.cpu().numpy()
                batch_size = batch_features_np.shape[0]
                num_features = np.prod(batch_features_np.shape[1:])
                batch_features_np = batch_features_np.reshape(batch_size, num_features)
                
                features.append(batch_features_np)
                metadata.extend(zip(names, info))

        features = np.concatenate(features, axis=0)

        import pandas as pd
        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])

        feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)

        combined_df = pd.concat([metadata_df, features_df], axis=1)
        return combined_df
    
    
    else:
        print(f"Pooling method {pooling_method} not implemented in patch")
        return None 