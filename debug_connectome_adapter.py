"""
Debug script to test the synapse_sampling module directly
"""

import torch
import numpy as np
from synapse_sampling.synapse_sampling import sample_synapses

class DummyProcessor:
    """Simple processor for testing"""
    def __init__(self):
        pass
    
    def transform(self, image):
        """Convert image to tensor"""
        return torch.from_numpy(image).float().unsqueeze(0) / 255.0  # Add channel dimension

def test_direct_sampling():
    """Test the direct sampling functionality"""
    print("Testing direct sampling...")
    
    # Sample with dummy policy
    raw, mask = sample_synapses(batch_size=2, policy="dummy", verbose=True)
    
    print(f"Raw shape: {raw.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Check the first sample
    raw_sample = raw[0, 0]  # Shape should be (80, 80, 80)
    mask_sample = mask[0, 0]  # Shape should be (80, 80, 80)
    
    print(f"Raw sample shape: {raw_sample.shape}")
    print(f"Mask sample shape: {mask_sample.shape}")
    
    # Create a simple tensor transformation
    # Convert to PyTorch tensor
    raw_tensor = torch.from_numpy(raw_sample).float()
    mask_tensor = torch.from_numpy(mask_sample).float()
    
    print(f"Raw tensor shape: {raw_tensor.shape}")
    print(f"Mask tensor shape: {mask_tensor.shape}")
    
    # Reshape to (C, D, H, W) format
    raw_tensor = raw_tensor.unsqueeze(0)  # Add channel dimension
    print(f"Raw tensor shape after adding channel: {raw_tensor.shape}")
    
    # Add batch dimension to get (B, C, D, H, W)
    raw_tensor = raw_tensor.unsqueeze(0)
    print(f"Raw tensor shape after adding batch: {raw_tensor.shape}")
    
    print("\nTest completed successfully!")

def test_manual_processing():
    """Test manual processing of the sampled data"""
    print("\nTesting manual processing...")
    
    # Sample with dummy policy
    raw, mask = sample_synapses(batch_size=1, policy="dummy", verbose=True)
    
    # Extract the first sample
    raw_vol = raw[0, 0]  # Shape: (80, 80, 80)
    mask_vol = mask[0, 0]  # Shape: (80, 80, 80)
    
    print(f"Raw volume shape: {raw_vol.shape}")
    print(f"Mask volume shape: {mask_vol.shape}")
    
    # Create a normalized copy of the raw volume
    raw_norm = raw_vol.astype(np.float32)
    min_val = raw_norm.min()
    max_val = raw_norm.max()
    if max_val > min_val:
        raw_norm = (raw_norm - min_val) / (max_val - min_val)
    
    # Apply the mask with alpha blending
    # Where mask is 1, blend with red color
    alpha = 1.0
    D, H, W = raw_norm.shape
    overlaid = np.zeros((D, H, W), dtype=np.float32)
    for d in range(D):
        slice_raw = raw_norm[d]
        slice_mask = mask_vol[d] > 0
        
        # Create the overlaid slice
        overlaid_slice = slice_raw.copy()
        overlaid_slice[slice_mask] = slice_raw[slice_mask] * (1 - alpha) + alpha
        overlaid[d] = overlaid_slice
    
    # Convert to uint8 for PIL compatibility
    overlaid_uint8 = (overlaid * 255).astype(np.uint8)
    
    # Extract frames and process them
    frames = [overlaid_uint8[d] for d in range(D)]
    
    print(f"Number of frames: {len(frames)}")
    print(f"First frame shape: {frames[0].shape}")
    
    # Process frames with a dummy processor
    processor = DummyProcessor()
    processed_frames = []
    for frame in frames:
        # Convert to tensor directly
        processed_frame = processor.transform(frame)
        processed_frames.append(processed_frame)
    
    # Stack frames along the depth dimension
    # This will give us (D, C, H, W)
    pixel_values = torch.stack(processed_frames)
    
    print(f"Pixel values shape after stacking: {pixel_values.shape}")
    
    # Reshape to (C, D, H, W) as expected by the model
    pixel_values = pixel_values.permute(1, 0, 2, 3)
    
    print(f"Pixel values shape after permutation: {pixel_values.shape}")
    
    # Add batch dimension to get (B, C, D, H, W)
    pixel_values = pixel_values.unsqueeze(0)
    
    print(f"Pixel values shape after adding batch: {pixel_values.shape}")
    
    print("\nManual processing test completed successfully!")

if __name__ == "__main__":
    test_direct_sampling()
    test_manual_processing() 