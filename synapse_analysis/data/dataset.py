import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from synapse_analysis.utils.processing import create_segmented_cube

class SynapseDataset(Dataset):
    def __init__(self, vol_data_dict, synapse_df, processor,
                 segmentation_type=6, subvol_size=80, num_frames=16,
                 alpha=0.3, fixed_samples=None):
        self.vol_data_dict = vol_data_dict
        
        # Filter the synapse_df to include only fixed samples if provided
        if fixed_samples:
            # Merge fixed_samples with synapse_df to only keep rows that are in fixed_samples
            fixed_samples_df = pd.DataFrame(fixed_samples)
            self.synapse_df = synapse_df.merge(fixed_samples_df, on=['Var1', 'bbox_name'], how='inner')
        else:
            self.synapse_df = synapse_df.reset_index(drop=True)
            
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        
        if raw_vol is None:
            return torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), 
                             dtype=torch.float32), syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), 
                        int(syn_info['central_coord_2']), 
                        int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), 
                      int(syn_info['side_1_coord_2']), 
                      int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), 
                      int(syn_info['side_2_coord_2']), 
                      int(syn_info['side_2_coord_3']))

        overlaid_cube = create_segmented_cube(
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
        )
        
        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        inputs = self.processor(frames, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0).float(), syn_info, bbox_name 