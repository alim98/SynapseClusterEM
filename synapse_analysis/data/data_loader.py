import os
import glob
import numpy as np
import pandas as pd
import imageio.v3 as iio
from typing import Tuple, Dict, Optional, List, Union
from torchvision import transforms
import torch
from .sample_normalization import apply_sample_normalization

def apply_global_normalization(tensor_data: Union[torch.Tensor, np.ndarray], 
                              global_stats: Dict[str, List[float]]) -> torch.Tensor:
    """
    Apply global normalization to tensor data.
    
    This function is deprecated and will be removed in a future version.
    It now falls back to standard normalization with default values.
    
    This function can be used independently in the data processing pipeline
    to apply global normalization to any tensor data.
    
    Args:
        tensor_data: Input tensor data to normalize [B, C, ...] or [C, ...]
        global_stats: Dictionary containing 'mean' and 'std' values
        
    Returns:
        Normalized tensor data
    """
    import warnings
    warnings.warn(
        "apply_global_normalization is deprecated and will be removed in a future version. "
        "Using standard normalization instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    
    # Convert to tensor if numpy array
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data)
    
    # Use default normalization values
    mean = torch.tensor([0.485])
    std = torch.tensor([0.229])
    
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
                 apply_global_norm=False, global_stats=None, apply_sample_norm=False):
        """
        Initialize the Synapse3DProcessor for image processing and normalization.
        
        Args:
            size: Target size for the images (height, width)
            mean: Mean for normalization (per channel)
            std: Standard deviation for normalization (per channel)
            apply_global_norm: Whether to apply global normalization (deprecated, always False)
            global_stats: Dictionary containing global 'mean' and 'std' if apply_global_norm is True (deprecated)
            apply_sample_norm: Whether to apply sample-wise normalization
        """
        self.size = size
        self.mean = mean
        self.std = std
        # Global normalization is disabled
        self.apply_global_norm = False
        self.global_stats = None
        self.apply_sample_norm = apply_sample_norm

        # Base transform without normalization
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # Full transform with normalization (always using standard normalization)
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
            # Use regular transform with per-frame normalization
            processed_frames = [self.transform(frame) for frame in frames]
            
        pixel_values = torch.stack(processed_frames)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

    def normalize_tensor(self, tensor, use_global_norm=None, use_sample_norm=None):
        """
        Apply normalization to a tensor.
        
        Args:
            tensor: Input tensor to normalize
            use_global_norm: Whether to use global normalization (deprecated, always False)
            use_sample_norm: Whether to use sample-wise normalization (overrides instance setting)
            
        Returns:
            Normalized tensor
        """
        # Determine normalization type
        apply_sample = self.apply_sample_norm if use_sample_norm is None else use_sample_norm
        
        # Sample-wise normalization
        if apply_sample:
            # Use sample-wise normalization
            return apply_sample_normalization(tensor)
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
        Factory method to create processor with standard normalization.
        This method is kept for backward compatibility but no longer uses global normalization.
        
        Args:
            data_loader: DataLoader (not used for normalization)
            size: Size for the images
            num_samples: Number of samples (not used)
            
        Returns:
            Processor instance with standard normalization applied
        """
        print("Warning: Global normalization is deprecated. Using standard normalization instead.")
        return cls(
            size=size,
            mean=(0.485,),
            std=(0.229,),
            apply_global_norm=False,
            global_stats=None
        )
        
    @classmethod
    def create_with_global_norm_from_volumes(cls, vol_data_dict, size=(80, 80)):
        """
        Factory method to create processor with standard normalization.
        This method is kept for backward compatibility but no longer uses global normalization.
        
        Args:
            vol_data_dict: Dictionary mapping bbox names to (raw_vol, seg_vol, add_mask_vol) tuples (not used for normalization)
            size: Size for the images
            
        Returns:
            Processor instance with standard normalization applied
        """
        print("Warning: Global normalization is deprecated. Using standard normalization instead.")
        return cls(
            size=size,
            mean=(0.485,),
            std=(0.229,),
            apply_global_norm=False,
            global_stats=None
        )

    @classmethod
    def create_with_standard_norm(cls, size=(80, 80), mean=(0.485,), std=(0.229,)):
        """
        Factory method to create processor with standard normalization (no global normalization).
        
        Args:
            size: Size for the images
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Processor instance with standard normalization applied
        """
        return cls(
            size=size,
            mean=mean,
            std=std,
            apply_global_norm=False,
            global_stats=None
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

def normalize_raw_data_before_segmentation(raw_data, size=(80, 80), mean=(0.485,), std=(0.229,)):
    """
    Normalize raw data before any segmentation step, without using global normalization.
    
    Args:
        raw_data: Raw image data as a list of frames or 3D volume
        size: Target size for the processed images
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized raw data ready for segmentation
    """
    # Create processor with standard normalization (no global normalization)
    processor = Synapse3DProcessor.create_with_standard_norm(
        size=size,
        mean=mean,
        std=std
    )
    
    # Convert to list of frames if the input is a 3D volume
    if isinstance(raw_data, np.ndarray) and raw_data.ndim == 3:
        frames = [raw_data[i] for i in range(raw_data.shape[0])]
    else:
        frames = raw_data
    
    # Apply normalization to raw data
    normalized_data = processor(frames)
    
    return normalized_data 