import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from newdl.dataloader2 import SynapseDataLoader

class SynapseDataset(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 80,
                 alpha: float = 0.3, normalize_across_volume: bool = True, 
                 smart_crop: bool = False, presynapse_weight: float = 0.5):
        self.vol_data_dict = vol_data_dict
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha
        self.data_loader = None
        self.normalize_across_volume = normalize_across_volume
        self.smart_crop = smart_crop
        self.presynapse_weight = presynapse_weight
        # Ensure the processor's normalization setting matches
        if hasattr(self.processor, 'normalize_volume'):
            self.processor.normalize_volume = normalize_across_volume

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        if raw_vol is None:
            return torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32), syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        if self.data_loader is None:
            self.data_loader = SynapseDataLoader("", "", "")
            
        overlaid_cube = self.data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.subvol_size,
            alpha=self.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=self.normalize_across_volume,
            smart_crop=self.smart_crop,
            presynapse_weight=self.presynapse_weight,
        )
        
        # Extract frames from the overlaid cube
        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        
        # Ensure we have the correct number of frames
        if len(frames) < self.num_frames:
            # Duplicate the last frame to reach the desired number
            frames += [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            # Sample frames evenly across the volume
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process frames and get pixel values
        inputs = self.processor(frames, return_tensors="pt")
        
        return inputs["pixel_values"].squeeze(0).float(), syn_info, bbox_name

class SynapseDataset2(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 16,
                 alpha: float = 0.3, fixed_samples=None, normalize_across_volume: bool = True,
                 smart_crop: bool = False, presynapse_weight: float = 0.5):
        self.vol_data_dict = vol_data_dict

        if fixed_samples:
            fixed_samples_df = pd.DataFrame(fixed_samples)
            self.synapse_df = synapse_df.merge(fixed_samples_df, on=['Var1', 'bbox_name'], how='inner')
        else:
            self.synapse_df = synapse_df.reset_index(drop=True)

        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha
        self.data_loader = None
        self.normalize_across_volume = normalize_across_volume
        self.smart_crop = smart_crop
        self.presynapse_weight = presynapse_weight
        # Ensure the processor's normalization setting matches
        if hasattr(self.processor, 'normalize_volume'):
            self.processor.normalize_volume = normalize_across_volume

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        if raw_vol is None:
            return torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32), syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        if self.data_loader is None:
            self.data_loader = SynapseDataLoader("", "", "")
            
        overlaid_cube = self.data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.subvol_size,
            alpha=self.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=self.normalize_across_volume,
            smart_crop=self.smart_crop,
            presynapse_weight=self.presynapse_weight,
        )
        
        # Extract frames from the overlaid cube
        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        
        # Ensure we have the correct number of frames
        if len(frames) < self.num_frames:
            # Duplicate the last frame to reach the desired number
            frames += [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            # Sample frames evenly across the volume
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process frames and get pixel values
        inputs = self.processor(frames, return_tensors="pt")
        
        return inputs["pixel_values"].squeeze(0).float(), syn_info, bbox_name 