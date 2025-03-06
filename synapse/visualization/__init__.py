"""
Visualization utilities for synapse data.

This module contains functions for visualizing synapse data, including
3D volumes, segmentation masks, and analysis results.
"""

from synapse.visualization.sample_fig import (
    create_gif_from_volume,
    visualize_specific_sample,
    visualize_all_samples_from_bboxes
)

__all__ = [
    'create_gif_from_volume',
    'visualize_specific_sample',
    'visualize_all_samples_from_bboxes'
] 