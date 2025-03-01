import os
import glob
import numpy as np
import pandas as pd
import imageio.v3 as iio
from typing import Tuple, Dict, Optional, List, Union
from torchvision import transforms
import torch

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
            processed_frames = [self.transform(frame) for frame in frames]
            
        pixel_values = torch.stack(processed_frames)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

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
        Calculate global mean and standard deviation directly from raw volumes.
        
        Args:
            vol_data_dict: Dictionary mapping bbox names to (raw_vol, seg_vol, add_mask_vol) tuples
            
        Returns:
            Dictionary with global 'mean' and 'std' values
        """
        # Initialize variables for statistics calculation
        sum_values = 0
        sum_squared = 0
        count = 0
        
        # Process each volume
        for bbox_name, (raw_vol, _, _) in vol_data_dict.items():
            # Update statistics
            sum_values += np.sum(raw_vol)
            sum_squared += np.sum(raw_vol.astype(np.float64) ** 2)
            count += raw_vol.size
        
        # Calculate mean and std
        mean = sum_values / count
        var = (sum_squared / count) - (mean ** 2)
        std = np.sqrt(var)
        
        global_stats = {
            'mean': [float(mean)],  # Wrap in list for compatibility with torchvision transforms
            'std': [float(std)]
        }
        
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
    add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
    
    if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
        return None, None, None
        
    try:
        raw_vol = np.stack([iio.imread(f) for f in raw_tif_files], axis=0)
        seg_vol = np.stack([iio.imread(f).astype(np.uint32) for f in seg_tif_files], axis=0)
        add_mask_vol = np.stack([iio.imread(f).astype(np.uint32) for f in add_mask_tif_files], axis=0)
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