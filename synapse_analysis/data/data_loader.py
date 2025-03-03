import os
import glob
import numpy as np
import pandas as pd
import imageio.v3 as iio
from typing import Tuple, Dict, Optional, List, Union
from torchvision import transforms
import torch

def apply_global_normalization(tensor_data: Union[torch.Tensor, np.ndarray], 
                              global_stats: Dict[str, List[float]]) -> torch.Tensor:
    """
    Apply global normalization to tensor data.
    
    This function can be used independently in the data processing pipeline
    to apply global normalization to any tensor data.
    
    Args:
        tensor_data: Input tensor data to normalize [B, C, ...] or [C, ...]
        global_stats: Dictionary containing 'mean' and 'std' values
        
    Returns:
        Normalized tensor data
    """
    # Convert to tensor if numpy array
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data)
    
    # Get global mean and std
    mean = torch.tensor(global_stats['mean'])
    std = torch.tensor(global_stats['std'])
    
    # Adjust mean and std if the input tensor is in [0,1] range but global stats are for [0,255] range
    min_val = tensor_data.min()
    max_val = tensor_data.max()
    
    # If tensor_data is in [0,1] range but global stats are for [0,255] range
    if max_val <= 1.0 and min_val >= 0.0 and mean[0] > 1.0:
        # Scale mean and std to [0,1] range
        mean = mean / 255.0
        std = std / 255.0
        print(f"Adjusted normalization parameters to [0,1] range: mean={mean.item()}, std={std.item()}")
    
    # Reshape mean and std based on input dimensions
    if tensor_data.dim() == 5:  # [B, C, D, H, W]
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)
    elif tensor_data.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor_data.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    # Apply normalization
    normalized_data = (tensor_data - mean) / std
    
    return normalized_data

class Synapse3DProcessor:
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,), 
                 apply_global_norm=False, global_stats=None):
        """
        Initialize the Synapse3DProcessor for image processing and normalization.
        
        Args:
            size: Target size for the images (height, width)
            mean: Mean for normalization (per channel)
            std: Standard deviation for normalization (per channel)
            apply_global_norm: Whether to apply global normalization
            global_stats: Dictionary containing global 'mean' and 'std' if apply_global_norm is True
        """
        self.size = size
        self.mean = mean
        self.std = std
        self.apply_global_norm = apply_global_norm
        self.global_stats = global_stats

        # Base transform without normalization
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # Full transform with normalization
        if apply_global_norm and global_stats:
            # Use global stats if provided
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=global_stats['mean'], std=global_stats['std']),
            ])
        else:
            # Use default stats
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, frames, return_tensors=None, skip_normalization=False):
        """
        Process frames with specified transformations.
        
        Args:
            frames: List of frames to process
            return_tensors: Whether to return tensors
            skip_normalization: Whether to skip normalization (useful when calculating global stats)
        
        Returns:
            Processed frames as tensors
        """
        if skip_normalization:
            processed_frames = [self.base_transform(frame) for frame in frames]
        else:
            # For global normalization, use base transform first (without normalization)
            # then apply global normalization once on the entire batch
            if self.apply_global_norm and self.global_stats:
                processed_frames = [self.base_transform(frame) for frame in frames]
                pixel_values = torch.stack(processed_frames)
                
                # Apply global normalization once to the entire batch
                mean = torch.tensor(self.global_stats['mean']).view(1, 1, 1).to(pixel_values.device)
                std = torch.tensor(self.global_stats['std']).view(1, 1, 1).to(pixel_values.device)
                
                # Adjust mean and std if the input tensor is in [0,1] range but global stats are for [0,255] range
                if pixel_values.max() <= 1.0 and mean.item() > 1.0:
                    # Scale mean and std to [0,1] range
                    mean = mean / 255.0
                    std = std / 255.0
                
                pixel_values = (pixel_values - mean) / std
                
                if return_tensors == "pt":
                    return {"pixel_values": pixel_values}
                else:
                    return pixel_values
            else:
                # Use regular transform with per-frame normalization
                processed_frames = [self.transform(frame) for frame in frames]
            
        pixel_values = torch.stack(processed_frames)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

    def normalize_tensor(self, tensor, use_global_norm=None):
        """
        Apply normalization to a tensor.
        
        Args:
            tensor: Input tensor to normalize
            use_global_norm: Whether to use global normalization (overrides instance setting)
            
        Returns:
            Normalized tensor
        """
        # Determine whether to use global normalization
        apply_global = self.apply_global_norm if use_global_norm is None else use_global_norm
        
        if apply_global and self.global_stats:
            # Use global normalization
            return apply_global_normalization(tensor, self.global_stats)
        else:
            # Use standard normalization
            if tensor.dim() == 5:  # [B, C, D, H, W]
                mean = torch.tensor(self.mean).view(1, -1, 1, 1, 1).to(tensor.device)
                std = torch.tensor(self.std).view(1, -1, 1, 1, 1).to(tensor.device)
            elif tensor.dim() == 4:  # [B, C, H, W]
                mean = torch.tensor(self.mean).view(1, -1, 1, 1).to(tensor.device)
                std = torch.tensor(self.std).view(1, -1, 1, 1).to(tensor.device)
            elif tensor.dim() == 3:  # [C, H, W]
                mean = torch.tensor(self.mean).view(-1, 1, 1).to(tensor.device)
                std = torch.tensor(self.std).view(-1, 1, 1).to(tensor.device)
            else:
                raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
                
            return (tensor - mean) / std

    @staticmethod
    def calculate_global_stats(data_loader, num_samples=None):
        """
        Calculate global mean and standard deviation across the dataset.
        
        Args:
            data_loader: DataLoader containing the dataset
            num_samples: Number of samples to use (None for all samples)
            
        Returns:
            Dictionary with global 'mean' and 'std' values
        """
        # Create processor with no normalization for this calculation
        temp_processor = Synapse3DProcessor()
        
        # Collect samples
        all_samples = []
        sample_count = 0
        
        for data in data_loader:
            if num_samples is not None and sample_count >= num_samples:
                break
                
            inputs = data[0]  # Assuming first element is the input tensor
            all_samples.append(inputs)
            sample_count += inputs.shape[0]
            
        # Concatenate all samples
        all_data = torch.cat(all_samples, dim=0)
        
        # Calculate mean and std over all dimensions except channel dimension
        global_mean = torch.mean(all_data, dim=[0, 2, 3]).tolist()
        global_std = torch.std(all_data, dim=[0, 2, 3]).tolist()
        
        return {
            'mean': global_mean,
            'std': global_std
        }

    @staticmethod
    def calculate_global_stats_from_volumes(vol_data_dict):
        """
        Calculate global mean and standard deviation from volumes.
        
        Args:
            vol_data_dict: Dictionary of volumes in the format {bbox_name: (raw_vol, seg_vol, combined_vol)}
            
        Returns:
            Dictionary with global 'mean' and 'std' values
        """
        # Initialize variables for statistics calculation
        all_values = []
        
        # Collect all raw volume data
        for bbox_name, (raw_vol, _, _) in vol_data_dict.items():
            # Print the data range of the volume for debugging
            print(f"Raw volume {bbox_name} range: min={raw_vol.min()}, max={raw_vol.max()}, dtype={raw_vol.dtype}")
            
            # Convert to float32 and flatten
            flat_vol = raw_vol.astype(np.float32).flatten()
            all_values.append(flat_vol)
        
        # Check if we have any valid data
        if not all_values:
            print("Warning: No valid volumes found. Using default normalization values.")
            return {
                'mean': [0.0],  # Default mean
                'std': [1.0]    # Default std
            }
        
        # Concatenate all values
        all_data = np.concatenate(all_values)
        
        # Calculate global mean and std
        mean = np.mean(all_data)
        std = np.std(all_data)
        
        if std == 0:
            std = 1.0  # Avoid division by zero
        
        global_stats = {
            'mean': [float(mean)],  # Wrap in list for compatibility with torchvision transforms
            'std': [float(std)]
        }
        
        print(f"Calculated global stats - mean: {mean}, std: {std}")
        return global_stats

    @classmethod
    def create_with_global_norm(cls, data_loader, size=(80, 80), num_samples=None):
        """
        Factory method to create processor with global normalization.
        
        Args:
            data_loader: DataLoader to calculate global stats from
            size: Size for the images
            num_samples: Number of samples to use for calculating stats
            
        Returns:
            Processor instance with global normalization applied
        """
        # Calculate global stats
        global_stats = cls.calculate_global_stats(data_loader, num_samples)
        
        # Create processor with global normalization
        return cls(
            size=size,
            mean=global_stats['mean'],
            std=global_stats['std'],
            apply_global_norm=True,
            global_stats=global_stats
        )
        
    @classmethod
    def create_with_global_norm_from_volumes(cls, vol_data_dict, size=(80, 80)):
        """
        Factory method to create processor with global normalization calculated directly from volumes.
        
        Args:
            vol_data_dict: Dictionary mapping bbox names to (raw_vol, seg_vol, add_mask_vol) tuples
            size: Size for the images
            
        Returns:
            Processor instance with global normalization applied
        """
        # Calculate global stats from volumes
        global_stats = cls.calculate_global_stats_from_volumes(vol_data_dict)
        
        # Create processor with global normalization
        return cls(
            size=size,
            mean=global_stats['mean'],
            std=global_stats['std'],
            apply_global_norm=True,
            global_stats=global_stats
        )

def load_volumes(bbox_name: str, raw_base_dir: str, seg_base_dir: str, 
                add_mask_base_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load raw, segmentation, and additional mask volumes for a given bbox.
    
    Args:
        bbox_name: Name of the bounding box
        raw_base_dir: Base directory for raw data
        seg_base_dir: Base directory for segmentation data
        add_mask_base_dir: Base directory for additional mask data
        
    Returns:
        Tuple of (raw_vol, seg_vol, add_mask_vol) or (None, None, None) if loading fails
    """
    raw_dir = os.path.join(raw_base_dir, bbox_name)
    seg_dir = os.path.join(seg_base_dir, bbox_name)
    
    if bbox_name.startswith("bbox"):
        bbox_num = bbox_name.replace("bbox", "")
        add_mask_dir = os.path.join(add_mask_base_dir, f"bbox_{bbox_num}")
    else:
        add_mask_dir = os.path.join(add_mask_base_dir, bbox_name)
        
    raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
    seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
    
    # Make additional mask files optional
    if add_mask_base_dir and os.path.exists(add_mask_dir):
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
    else:
        add_mask_tif_files = []
    
    # Check if raw and segmentation files exist
    if not raw_tif_files or not seg_tif_files:
        print(f"Missing raw or segmentation files for {bbox_name}")
        return None, None, None
        
    try:
        raw_vol = np.stack([iio.imread(f) for f in raw_tif_files], axis=0)
        seg_vol = np.stack([iio.imread(f).astype(np.uint32) for f in seg_tif_files], axis=0)
        
        # Only load additional mask if files exist
        if add_mask_tif_files:
            add_mask_vol = np.stack([iio.imread(f).astype(np.uint32) for f in add_mask_tif_files], axis=0)
        else:
            # Create a dummy mask of zeros with the same shape as raw_vol
            add_mask_vol = np.zeros_like(raw_vol, dtype=np.uint32)
            print(f"No additional mask files found for {bbox_name}. Using dummy mask.")
            
        return raw_vol, seg_vol, add_mask_vol
    except Exception as e:
        print(f"Error loading volumes for {bbox_name}: {str(e)}")
        return None, None, None

def load_synapse_data(bbox_names: list, excel_dir: str) -> pd.DataFrame:
    """
    Load synapse data from Excel files for given bounding boxes.
    
    Args:
        bbox_names: List of bounding box names
        excel_dir: Directory containing Excel files
        
    Returns:
        DataFrame containing combined synapse data
    """
    dfs = []
    for bbox in bbox_names:
        excel_path = os.path.join(excel_dir, f"{bbox}.xlsx")
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            df['bbox_name'] = bbox
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid Excel files found")
        
    return pd.concat(dfs, ignore_index=True)

def load_all_volumes(bbox_names: list, raw_base_dir: str, seg_base_dir: str, 
                    add_mask_base_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load volumes for all specified bounding boxes.
    
    Args:
        bbox_names: List of bounding box names
        raw_base_dir: Base directory for raw data
        seg_base_dir: Base directory for segmentation data
        add_mask_base_dir: Base directory for additional mask data
        
    Returns:
        Dictionary mapping bbox names to their volume data
    """
    vol_data_dict = {}
    for bbox_name in bbox_names:
        volumes = load_volumes(bbox_name, raw_base_dir, seg_base_dir, add_mask_base_dir)
        if all(v is not None for v in volumes):
            vol_data_dict[bbox_name] = volumes
    return vol_data_dict

def calculate_global_stats(raw_base_dir, seg_base_dir, add_mask_base_dir, excel_dir, 
                     segmentation_types=None, bbox_names=None, num_samples=100):
    """
    Calculate global mean and standard deviation for normalization across all volumes.
    
    Args:
        raw_base_dir: Base directory for raw data
        seg_base_dir: Base directory for segmentation data
        add_mask_base_dir: Base directory for additional mask data
        excel_dir: Directory containing Excel files
        segmentation_types: List of segmentation types to use
        bbox_names: List of bounding box names
        num_samples: Number of samples to use for statistics
        
    Returns:
        Dictionary with global 'mean' and 'std' values
    """
    # Load volumes
    vol_data_dict = load_all_volumes(bbox_names, raw_base_dir, seg_base_dir, add_mask_base_dir)
    
    # Use the class method to calculate stats from volumes
    return Synapse3DProcessor.calculate_global_stats_from_volumes(vol_data_dict) 