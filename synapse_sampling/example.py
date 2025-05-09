"""
Example script demonstrating how to use the synapse_sampling module directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os

# Add parent directory to path so Python can find the module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import synapse_sampling module
from synapse_sampling.synapse_sampling import sample_synapses

def visualize_synapse_sample(raw, mask, sample_idx=0, slice_idx=None):
    """
    Visualize a synapse sample with the mask overlay.
    
    Args:
        raw (np.ndarray): Raw volume data with shape (batch_size, 1, 80, 80, 80)
        mask (np.ndarray): Mask volume data with shape (batch_size, 1, 80, 80, 80)
        sample_idx (int): Index of the sample to visualize
        slice_idx (int, optional): Index of the slice to visualize. If None, use middle slice.
    """
    # Extract single sample
    raw_vol = raw[sample_idx, 0]  # Shape (80, 80, 80)
    mask_vol = mask[sample_idx, 0]  # Shape (80, 80, 80)
    
    # Use middle slice if not specified
    if slice_idx is None:
        slice_idx = raw_vol.shape[2] // 2
    
    # Extract slice
    raw_slice = raw_vol[:, :, slice_idx]
    mask_slice = mask_vol[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot raw data
    im1 = axes[0].imshow(raw_slice, cmap='gray')
    axes[0].set_title(f"Raw (slice {slice_idx})")
    axes[0].axis('off')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    # Plot mask
    im2 = axes[1].imshow(mask_slice, cmap='binary')
    axes[1].set_title(f"Mask (slice {slice_idx})")
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # Plot overlay
    # Create colored overlay (red mask on grayscale image)
    overlay = np.zeros((*raw_slice.shape, 3))
    # Normalize raw to [0, 1]
    raw_norm = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min() + 1e-10)
    for c in range(3):
        overlay[:, :, c] = raw_norm
    # Add red mask with alpha blending
    alpha = 0.5
    overlay[:, :, 0][mask_slice > 0] = alpha * 1.0 + (1 - alpha) * raw_norm[mask_slice > 0]
    overlay[:, :, 1][mask_slice > 0] = (1 - alpha) * raw_norm[mask_slice > 0]
    overlay[:, :, 2][mask_slice > 0] = (1 - alpha) * raw_norm[mask_slice > 0]
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (slice {slice_idx})")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    print("Sampling synapses from connectome...")
    # Sample 5 synapses with random policy
    raw, mask = sample_synapses(batch_size=5, policy="random", verbose=True)
    
    print(f"Raw data shape: {raw.shape}")
    print(f"Mask data shape: {mask.shape}")
    
    # Visualize each sample
    for i in range(raw.shape[0]):
        print(f"\nVisualizing sample {i+1}...")
        fig = visualize_synapse_sample(raw, mask, sample_idx=i)
        plt.savefig(f"sample_{i+1}.png", dpi=150)
        plt.close(fig)
        print(f"Saved as sample_{i+1}.png")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 