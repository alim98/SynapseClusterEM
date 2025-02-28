import os
import glob
import numpy as np
import pandas as pd
import imageio.v3 as iio
from typing import Tuple, Dict, Optional
from torchvision import transforms
import torch

class Synapse3DProcessor:
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __call__(self, frames, return_tensors=None):
        processed_frames = [self.transform(frame) for frame in frames]
        pixel_values = torch.stack(processed_frames)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

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