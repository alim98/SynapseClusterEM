import os
import glob
import numpy as np
import pandas as pd
import imageio.v3 as iio
from typing import Tuple, Dict, Optional, List, Union
from torchvision import transforms
import torch
from .sample_normalization import apply_sample_normalization
import json

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
                 apply_global_norm=False, global_stats=None, apply_sample_norm=False,
                 global_stats_path=None, auto_compute_stats=False, bbox_names=None,
                 raw_base_dir=None, seg_base_dir=None, add_mask_base_dir=None):
        """
        Initialize the Synapse3DProcessor for image processing and normalization.
        
        Args:
            size: Target size for the images (height, width)
            mean: Mean for normalization (per channel)
            std: Standard deviation for normalization (per channel)
            apply_global_norm: Whether to apply global normalization
            global_stats: Dictionary containing global 'mean' and 'std' if apply_global_norm is True
            apply_sample_norm: Whether to apply sample-wise normalization
            global_stats_path: Path to JSON file containing global stats
            auto_compute_stats: Whether to auto-compute global stats if not available
            bbox_names: List of bounding box names for auto-computing stats
            raw_base_dir: Base directory for raw data, for auto-computing stats
            seg_base_dir: Base directory for segmentation data, for auto-computing stats
            add_mask_base_dir: Base directory for additional mask data, for auto-computing stats
        """
        self.size = size
        self.mean = mean
        self.std = std
        
        # Global normalization settings
        self.apply_global_norm = apply_global_norm
        self.global_stats = global_stats
        self.global_stats_path = global_stats_path
        self.apply_sample_norm = apply_sample_norm
        
        # Auto-compute settings
        self.auto_compute_stats = auto_compute_stats
        self.bbox_names = bbox_names
        self.raw_base_dir = raw_base_dir
        self.seg_base_dir = seg_base_dir
        self.add_mask_base_dir = add_mask_base_dir
        
        # Load or compute global stats if needed
        if self.apply_global_norm:
            self._initialize_global_stats()

        # Base transform without normalization
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # Setup transforms based on normalization settings
        self._setup_transforms()
        
    def _initialize_global_stats(self):
        """Initialize global stats by loading from file or computing."""
        if self.global_stats is not None:
            # Already have global stats
            print("Using provided global stats.")
            return
            
        if self.global_stats_path:
            # Try to load from file
            try:
                with open(self.global_stats_path, 'r') as f:
                    self.global_stats = json.load(f)
                print(f"Loaded global stats from {self.global_stats_path}")
                self.mean = self.global_stats.get('mean')
                self.std = self.global_stats.get('std')
                return
            except Exception as e:
                print(f"Error loading global stats from {self.global_stats_path}: {e}")
                
        # Stats not provided and couldn't load from file
        if self.auto_compute_stats and self.bbox_names and self.raw_base_dir:
            print("Auto-computing global normalization stats...")
            calculator = GlobalNormalizationCalculator(
                raw_base_dir=self.raw_base_dir,
                output_file=self.global_stats_path
            )
            self.global_stats = calculator.compute_from_volumes(
                bbox_names=self.bbox_names,
                seg_base_dir=self.seg_base_dir,
                add_mask_base_dir=self.add_mask_base_dir
            )
            self.mean = self.global_stats.get('mean')
            self.std = self.global_stats.get('std')
            print(f"Auto-computed global stats - mean: {self.mean}, std: {self.std}")
        else:
            print("Global normalization requested but no stats available. Using default values.")
            # Keep using the default values
            
    def _setup_transforms(self):
        """Setup the transformation pipeline based on normalization settings."""
        # Full transform with normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
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
            # Apply only base transform without any normalization
            processed_frames = [self.base_transform(frame) for frame in frames]
            pixel_values = torch.stack(processed_frames)
        else:
            # Apply base transform and then normalization
            base_frames = [self.base_transform(frame) for frame in frames]
            pixel_values = torch.stack(base_frames)
            
            # Apply normalization using normalize_tensor method
            pixel_values = self.normalize_tensor(pixel_values)
            
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

    def normalize_tensor(self, tensor, use_global_norm=None, use_sample_norm=None):
        """
        Apply normalization to a tensor.
        
        Args:
            tensor: Input tensor to normalize
            use_global_norm: Whether to use global normalization
            use_sample_norm: Whether to use sample-wise normalization (overrides instance setting)
            
        Returns:
            Normalized tensor
        """
        # Determine normalization types
        apply_global = self.apply_global_norm if use_global_norm is None else use_global_norm
        apply_sample = self.apply_sample_norm if use_sample_norm is None else use_sample_norm
        
        # First apply global normalization if required
        if apply_global and self.global_stats:
            # If global stats available, use them
            global_mean = torch.tensor(self.global_stats['mean'], device=tensor.device)
            global_std = torch.tensor(self.global_stats['std'], device=tensor.device)
            # Reshape for broadcasting
            if len(tensor.shape) == 4:  # (N, C, H, W)
                global_mean = global_mean.view(1, -1, 1, 1)
                global_std = global_std.view(1, -1, 1, 1)
            elif len(tensor.shape) == 3:  # (C, H, W)
                global_mean = global_mean.view(-1, 1, 1)
                global_std = global_std.view(-1, 1, 1)
            
            # Apply normalization
            tensor = (tensor - global_mean) / global_std
        elif apply_global:
            # Global normalization requested but no stats - use standard normalization
            return transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        
        # Then apply sample normalization if required
        if apply_sample:
            tensor = apply_sample_normalization(tensor)
            
        return tensor

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
        # temp_processor = Synapse3DProcessor()
        
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
    def from_config(cls, config):
        """
        Create a Synapse3DProcessor from config parameters.
        
        Args:
            config: Dictionary or object with configuration parameters
                Required: size
                Optional: use_global_norm, global_stats_path, apply_sample_norm,
                          auto_compute_stats, bbox_names, raw_base_dir, seg_base_dir,
                          add_mask_base_dir
        
        Returns:
            Synapse3DProcessor instance configured according to parameters
        """
        # Get required parameters
        size = getattr(config, 'size', (80, 80))
        
        # Get optional parameters
        apply_global_norm = getattr(config, 'use_global_norm', False)
        global_stats_path = getattr(config, 'global_stats_path', None)
        apply_sample_norm = getattr(config, 'apply_sample_norm', False)
        auto_compute_stats = getattr(config, 'auto_compute_stats', False)
        
        # Get parameters for auto-computing stats if needed
        bbox_names = getattr(config, 'bbox_names', None)
        raw_base_dir = getattr(config, 'raw_base_dir', None)
        seg_base_dir = getattr(config, 'seg_base_dir', None)
        add_mask_base_dir = getattr(config, 'add_mask_base_dir', None)
        
        # Create processor
        return cls(
            size=size,
            apply_global_norm=apply_global_norm,
            global_stats_path=global_stats_path,
            apply_sample_norm=apply_sample_norm,
            auto_compute_stats=auto_compute_stats,
            bbox_names=bbox_names,
            raw_base_dir=raw_base_dir,
            seg_base_dir=seg_base_dir,
            add_mask_base_dir=add_mask_base_dir
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

class GlobalNormalizationCalculator:
    """
    A class for computing global normalization statistics from raw data.
    
    This class provides functionality to:
    1. Load raw volumes from specified directories
    2. Compute global statistics (mean, std, min, max)
    3. Save the computed statistics to a file
    4. Load pre-computed statistics from a file
    """
    
    def __init__(self, raw_base_dir=None, output_file=None):
        """
        Initialize the GlobalNormalizationCalculator.
        
        Args:
            raw_base_dir (str, optional): Base directory for raw data
            output_file (str, optional): File path to save computed statistics
        """
        self.raw_base_dir = raw_base_dir
        self.output_file = output_file
        self.stats = None
    
    def compute_from_volumes(self, bbox_names, seg_base_dir=None, add_mask_base_dir=None, verbose=True):
        """
        Compute global statistics from raw volumes.
        
        Args:
            bbox_names (list): List of bounding box names
            seg_base_dir (str, optional): Base directory for segmentation data (not used for stats)
            add_mask_base_dir (str, optional): Base directory for additional masks (not used for stats)
            verbose (bool): Whether to print detailed information during computation
            
        Returns:
            dict: Dictionary containing global statistics
        """
        if verbose:
            print(f"Computing global normalization stats for {len(bbox_names)} volumes...")
        
        # Load all volumes
        vol_data_dict = load_all_volumes(
            bbox_names, 
            self.raw_base_dir, 
            seg_base_dir or self.raw_base_dir, 
            add_mask_base_dir or self.raw_base_dir
        )
        
        # Use the existing method to calculate stats
        self.stats = Synapse3DProcessor.calculate_global_stats_from_volumes(vol_data_dict)
        
        # Add additional statistics
        all_values = []
        for bbox_name, (raw_vol, _, _) in vol_data_dict.items():
            flat_vol = raw_vol.astype(np.float32).flatten()
            all_values.append(flat_vol)
        
        if all_values:
            all_data = np.concatenate(all_values)
            self.stats['min'] = float(np.min(all_data))
            self.stats['max'] = float(np.max(all_data))
            self.stats['percentile_1'] = float(np.percentile(all_data, 1))
            self.stats['percentile_99'] = float(np.percentile(all_data, 99))
        
        if self.output_file and self.stats:
            self.save_stats()
            
        if verbose:
            self._print_stats()
            
        return self.stats
    
    def compute_from_dataloader(self, data_loader, num_samples=None, verbose=True):
        """
        Compute global statistics from a DataLoader.
        
        Args:
            data_loader: PyTorch DataLoader containing the dataset
            num_samples (int, optional): Number of samples to use (None for all)
            verbose (bool): Whether to print detailed information during computation
            
        Returns:
            dict: Dictionary containing global statistics
        """
        if verbose:
            print(f"Computing global normalization stats from dataloader...")
        
        # Use the existing method to calculate stats
        self.stats = Synapse3DProcessor.calculate_global_stats(data_loader, num_samples)
        
        if self.output_file and self.stats:
            self.save_stats()
            
        if verbose:
            self._print_stats()
            
        return self.stats
    
    def save_stats(self, output_file=None):
        """
        Save the computed statistics to a file.
        
        Args:
            output_file (str, optional): File path to save to (uses self.output_file if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.stats:
            print("No statistics to save.")
            return False
            
        out_file = output_file or self.output_file
        if not out_file:
            print("No output file specified.")
            return False
            
        try:
            with open(out_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
            print(f"Statistics saved to {out_file}")
            return True
        except Exception as e:
            print(f"Error saving statistics: {e}")
            return False
    
    def load_stats(self, input_file=None):
        """
        Load pre-computed statistics from a file.
        
        Args:
            input_file (str, optional): File path to load from (uses self.output_file if None)
            
        Returns:
            dict: Loaded statistics dictionary
        """
        in_file = input_file or self.output_file
        if not in_file:
            print("No input file specified.")
            return None
            
        try:
            with open(in_file, 'r') as f:
                self.stats = json.load(f)
            print(f"Statistics loaded from {in_file}")
            return self.stats
        except Exception as e:
            print(f"Error loading statistics: {e}")
            return None
    
    def _print_stats(self):
        """Print the computed statistics in a readable format."""
        if not self.stats:
            print("No statistics available.")
            return
            
        print("\nGlobal Normalization Statistics:")
        print("-" * 30)
        for key, value in self.stats.items():
            print(f"{key}: {value}")
        print("-" * 30)
    
    def get_normalization_parameters(self):
        """
        Get the mean and std for use with normalization.
        
        Returns:
            tuple: (mean, std) as expected by Synapse3DProcessor
        """
        if not self.stats:
            return None, None
            
        return self.stats.get('mean'), self.stats.get('std')
