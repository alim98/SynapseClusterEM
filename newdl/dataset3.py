import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from newdl.dataloader3 import SynapseDataLoader

class SynapseDataset(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 80,
                 alpha: float = 0.3, normalize_across_volume: bool = True, 
                 smart_crop: bool = True, presynapse_weight: float = 0.7,
                 normalize_presynapse_size: bool = True, target_percentage: float = None,
                 size_tolerance: float = 0.1):
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
        self.normalize_presynapse_size = normalize_presynapse_size
        self.target_percentage = target_percentage
        self.size_tolerance = size_tolerance
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
            print(f"Volume data not found for {bbox_name}. Returning None.")
            return None

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
            normalize_across_volume=True,
            smart_crop=self.smart_crop,
            presynapse_weight=self.presynapse_weight,
            normalize_presynapse_size=self.normalize_presynapse_size,
            target_percentage=self.target_percentage,
            size_tolerance=self.size_tolerance,
        )
        
        # Handle case when overlaid_cube is None (sample discarded)
        if overlaid_cube is None:
            print(f"Sample {bbox_name} was discarded during processing. Returning None instead of zeros.")
            return None
        
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

class Synapsedataset2(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 16,
                 alpha: float = 0.3, fixed_samples=None, normalize_across_volume: bool = True,
                 smart_crop: bool = False, presynapse_weight: float = 0.5,
                 normalize_presynapse_size: bool = False, target_percentage: float = None,
                 size_tolerance: float = 0.1):
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
        self.normalize_presynapse_size = normalize_presynapse_size
        self.target_percentage = target_percentage
        self.size_tolerance = size_tolerance
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
            print(f"Volume data not found for {bbox_name}. Returning None.")
            return None

        # If slice_number is provided in fixed_samples, use it
        slice_number = syn_info.get('slice_number', None)

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
            normalize_presynapse_size=self.normalize_presynapse_size,
            target_percentage=self.target_percentage,
            size_tolerance=self.size_tolerance,
        )
        
        # Handle case when overlaid_cube is None (sample discarded)
        if overlaid_cube is None:
            print(f"Sample {bbox_name} was discarded during processing. Returning None instead of zeros.")
            return None
        
        # Extract frames from the overlaid cube
        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        
        # If we have a specified slice_number, center frames around it
        if slice_number is not None:
            # Adjust to 0-based index
            slice_index = slice_number - 1 if slice_number > 0 else 0
            
            # Constrain to valid range
            slice_index = min(max(0, slice_index), len(frames) - 1)
            
            # Calculate the range of frames to include
            half_frames = self.num_frames // 2
            start_idx = max(0, slice_index - half_frames)
            end_idx = min(len(frames), slice_index + half_frames + (self.num_frames % 2))
            
            # If we're too close to the beginning or end, adjust to get the right number of frames
            if end_idx - start_idx < self.num_frames:
                if start_idx == 0:
                    end_idx = min(len(frames), self.num_frames)
                elif end_idx == len(frames):
                    start_idx = max(0, len(frames) - self.num_frames)
            
            # Select the frames
            frames = frames[start_idx:end_idx]
        
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