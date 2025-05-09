import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Any, Optional


from synapse_sampling.synapse_sampling import sample_synapses




class SynapseConnectomeAdapter:
    """
    Adapter class to integrate synapse_sampling module with the existing pipeline.
    This replaces the functionality of SynapseDataLoader, providing sampled data from connectome.
    """
    def __init__(self, batch_size=10, policy="random", verbose=False):
        """
        Initialize the adapter with sampling parameters
        """
        self.batch_size = batch_size
        self.policy = policy
        self.verbose = verbose
        
    def load_data(self) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
        """
        Load data using the synapse_sampling module instead of loading from files.
        
        Returns:
            Tuple[Dict, DataFrame]: A tuple containing:
                - Dictionary mapping sample IDs to (raw_vol, seg_vol, mask_vol) tuples
                - DataFrame with metadata for each sample
        """
        
        raw_volumes, mask_volumes = sample_synapses(batch_size=self.batch_size, policy=self.policy, verbose=self.verbose)
        
        
        vol_data_dict = {}
        
        
        metadata_list = []
        
        
        for i in range(raw_volumes.shape[0]):
            
            raw_vol = raw_volumes[i, 0]  
            mask_vol = mask_volumes[i, 0]  
            
            
            
            seg_vol = np.zeros_like(raw_vol)
            
            
            sample_id = f"sample_{i+1}"
            
            
            vol_data_dict[sample_id] = (raw_vol, seg_vol, mask_vol)
            
            
            
            center_x, center_y, center_z = raw_vol.shape[0]//2, raw_vol.shape[1]//2, raw_vol.shape[2]//2
            
            
            metadata = {
                'bbox_name': sample_id,
                'Var1': i,  
                'central_coord_1': center_x,
                'central_coord_2': center_y,
                'central_coord_3': center_z,
                
                
                'side_1_coord_1': center_x,
                'side_1_coord_2': center_y,
                'side_1_coord_3': center_z,
                'side_2_coord_1': center_x,
                'side_2_coord_2': center_y,
                'side_2_coord_3': center_z,
            }
            metadata_list.append(metadata)
        
        
        synapse_df = pd.DataFrame(metadata_list)
        
        return vol_data_dict, synapse_df


class ConnectomeDataset(Dataset):
    """
    Dataset class for connectome data that works with the existing pipeline.
    Similar to SynapseDataset but uses the adapter for data loading.
    """
    def __init__(self, processor, segmentation_type: int = 10, alpha: float = 1.0, 
                 batch_size: int = 10, policy: str = "random", verbose: bool = False):
        """
        Initialize the dataset with the adapter
        
        Args:
            processor: Processor for transforming images
            segmentation_type: Segmentation type to use (default 10)
            alpha: Alpha value for blending (default 1.0)
            batch_size: Number of samples to load
            policy: Sampling policy ("random" or "dummy")
            verbose: Whether to print verbose information
        """
        
        self.adapter = SynapseConnectomeAdapter(batch_size, policy, verbose)
        
        
        self.vol_data_dict, self.synapse_df = self.adapter.load_data()
        
        
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.alpha = alpha
        
        
        self.num_frames = 80  
        self.subvol_size = 80
    
    def __len__(self):
        return len(self.synapse_df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple: (pixel_values, metadata, sample_id)
        """
        
        syn_info = self.synapse_df.iloc[idx]
        sample_id = syn_info['bbox_name']
        
        
        raw_vol, _, mask_vol = self.vol_data_dict.get(sample_id, (None, None, None))
        
        if raw_vol is None:
            print(f"Volume data not found for {sample_id}. Returning None.")
            return None
        
        
        if idx == 0:  
            print(f"Raw volume shape: {raw_vol.shape}")
            print(f"Mask volume shape: {mask_vol.shape}")
        
        
        
        if len(raw_vol.shape) == 4 and raw_vol.shape[0] == 1:
            
            raw_vol = raw_vol[0]  
        
        if len(mask_vol.shape) == 4 and mask_vol.shape[0] == 1:
            
            mask_vol = mask_vol[0]  
        
        
        if idx == 0:  
            print(f"Raw volume shape after preprocessing: {raw_vol.shape}")
            print(f"Mask volume shape after preprocessing: {mask_vol.shape}")
        
        
        raw_norm = raw_vol.astype(np.float32)
        min_val = raw_norm.min()
        max_val = raw_norm.max()
        if max_val > min_val:
            raw_norm = (raw_norm - min_val) / (max_val - min_val)
        
        
        
        D, H, W = raw_norm.shape
        overlaid = np.zeros((D, H, W), dtype=np.float32)
        for d in range(D):
            slice_raw = raw_norm[d]
            slice_mask = mask_vol[d] > 0
            
            
            overlaid_slice = slice_raw.copy()
            overlaid_slice[slice_mask] = slice_raw[slice_mask] * (1 - self.alpha) + self.alpha
            overlaid[d] = overlaid_slice
        
        
        overlaid_uint8 = (overlaid * 255).astype(np.uint8)
        
        
        frames = [overlaid_uint8[d] for d in range(D)]
        
        
        if idx == 0:
            print(f"Number of frames: {len(frames)}")
            print(f"First frame shape: {frames[0].shape}")
        
        
        processed_frames = []
        for frame in frames:
            
            processed_frame = self.processor.transform(frame)
            processed_frames.append(processed_frame)
        
        
        
        pixel_values = torch.stack(processed_frames)
        
        
        if idx == 0:
            print(f"Pixel values shape after stacking: {pixel_values.shape}")
        
        
        
        pixel_values = pixel_values.permute(1, 0, 2, 3)
        
        
        if idx == 0:
            print(f"Pixel values shape after permutation: {pixel_values.shape}")
        
        return pixel_values, syn_info, sample_id 