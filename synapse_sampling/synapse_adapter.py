"""
Synapse Sampling Adapter for the Synapse Pipeline

This adapter integrates the synapse_sampling module with the existing pipeline
to enable sampling from the connectome.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List, Union, Optional
from tqdm import tqdm

from synapse_sampling.synapse_sampling import sample_synapses
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset

class SynapseSamplingDataset(torch.utils.data.Dataset):
    """
    Dataset class that uses synapse_sampling to generate samples on-the-fly
    """
    
    def __init__(self, 
                 batch_size: int = 100, 
                 policy: str = "dummy",  
                 processor = None,
                 segmentation_type: int = 10,  
                 alpha: float = 0.3,
                 verbose: bool = False):
        """
        Initialize the dataset with synapse_sampling parameters
        
        Args:
            batch_size: Number of samples to generate
            policy: Sampling policy ("dummy" or "random")
            processor: Image processor for the raw data
            segmentation_type: Type of segmentation (default: 10)
            alpha: Alpha value for blending mask with raw data
            verbose: Whether to show progress information
        """
        self.batch_size = batch_size
        self.policy = policy
        self.processor = processor if processor is not None else Synapse3DProcessor()
        self.segmentation_type = segmentation_type
        self.alpha = alpha
        self.verbose = verbose
        
        
        if self.verbose:
            print(f"Generating {batch_size} samples using {policy} policy...")
        self.raw_data, self.masks = sample_synapses(batch_size=batch_size, policy=policy, verbose=verbose)
        
        
        self.metadata = []
        for i in range(batch_size):
            self.metadata.append({
                'bbox_name': f'sample_{i}',
                'central_coord_1': 40,  
                'central_coord_2': 40,
                'central_coord_3': 40,
                'Var1': i  
            })
        
        if self.verbose:
            print(f"Generated {len(self.raw_data)} samples with shape {self.raw_data.shape}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        
        raw = self.raw_data[idx]
        mask = self.masks[idx]
        
        
        
        blended = raw * (1 - self.alpha) + mask * self.alpha * 255
        
        
        blended = blended / 255.0
        
        
        
        cube = np.transpose(blended, (1, 2, 0, 3))
        
        
        frames = [cube[..., z] for z in range(cube.shape[3])]
        
        
        if self.processor is not None:
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0).float()
        else:
            
            pixel_values = torch.tensor(np.stack(frames)).float()
        
        
        metadata = self.metadata[idx]
        
        return pixel_values, pd.Series(metadata), metadata['bbox_name']


class SynapseSamplingAdapter:
    """
    Adapter class that integrates synapse_sampling with the synapse pipeline
    """
    
    def __init__(self, config_obj=None):
        """
        Initialize the adapter with the given configuration
        
        Args:
            config_obj: Configuration object
        """
        self.config = config_obj
        
    def create_dataset(self, 
                       batch_size: int = 100, 
                       policy: str = "dummy",
                       segmentation_type: int = 10, 
                       alpha: float = 0.3,
                       processor = None, 
                       verbose: bool = False) -> SynapseSamplingDataset:
        """
        Create a dataset that uses synapse_sampling to generate samples
        
        Args:
            batch_size: Number of samples to generate
            policy: Sampling policy ("dummy" or "random")
            segmentation_type: Type of segmentation
            alpha: Alpha value for blending
            processor: Image processor for the raw data
            verbose: Whether to show progress information
            
        Returns:
            SynapseSamplingDataset: Dataset ready for use in the pipeline
        """
        return SynapseSamplingDataset(
            batch_size=batch_size,
            policy=policy,
            processor=processor,
            segmentation_type=segmentation_type,
            alpha=alpha,
            verbose=verbose
        )
    
    def load_and_prepare_data_for_pipeline(self, 
                                          batch_size: int = 100, 
                                          policy: str = "dummy",
                                          verbose: bool = False) -> Tuple[Dict, pd.DataFrame]:
        """
        Load and prepare data for the synapse pipeline, compatible with the existing interface
        
        Args:
            batch_size: Number of samples to generate
            policy: Sampling policy ("dummy" or "random")
            verbose: Whether to show progress information
            
        Returns:
            tuple: (vol_data_dict, synapse_df) - Compatible with the existing pipeline
        """
        
        raw_data, masks = sample_synapses(batch_size=batch_size, policy=policy, verbose=verbose)
        
        
        vol_data_dict = {}
        
        
        synapse_data = []
        
        for i in range(batch_size):
            bbox_name = f"sample_{i}"
            
            
            
            vol_data_dict[bbox_name] = (np.squeeze(raw_data[i]), np.squeeze(masks[i]), np.squeeze(masks[i]))
            
            
            synapse_data.append({
                'bbox_name': bbox_name,
                'Var1': i,
                
                'central_coord_1': 40,
                'central_coord_2': 40,
                'central_coord_3': 40,
                
                'side_1_coord_1': 20,
                'side_1_coord_2': 20,
                'side_1_coord_3': 20,
                'side_2_coord_1': 60,
                'side_2_coord_2': 60,
                'side_2_coord_3': 60,
            })
        
        
        synapse_df = pd.DataFrame(synapse_data)
        
        if verbose:
            print(f"Created synthetic vol_data_dict with {len(vol_data_dict)} samples")
            print(f"Created synthetic synapse_df with {len(synapse_df)} rows")
        
        return vol_data_dict, synapse_df



if __name__ == "__main__":
    
    adapter = SynapseSamplingAdapter()
    
    
    dataset = adapter.create_dataset(batch_size=10, policy="dummy", verbose=True)
    print(f"Dataset size: {len(dataset)}")
    
    
    pixel_values, metadata, bbox_name = dataset[0]
    print(f"Sample shape: {pixel_values.shape}")
    print(f"Sample bbox_name: {bbox_name}")
    
    
    vol_data_dict, synapse_df = adapter.load_and_prepare_data_for_pipeline(batch_size=10, policy="dummy", verbose=True)
    print(f"vol_data_dict size: {len(vol_data_dict)}")
    print(f"synapse_df size: {len(synapse_df)}") 