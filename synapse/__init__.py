"""
Synapse Analysis Package

A framework for 3D synapse data analysis, feature extraction and clustering.
"""

__version__ = '0.1.0'

# Import commonly used components
from synapse.models import Vgg3D, load_model_from_checkpoint
from synapse.data import SynapseDataset, SynapseDataset2, Synapse3DProcessor, SynapseDataLoader
from synapse.utils import config
from synapse.visualization import create_gif_from_volume, visualize_specific_sample, visualize_all_samples_from_bboxes
from synapse.clustering import (
    load_and_cluster_features,
    find_random_samples_in_clusters,
    find_closest_samples_in_clusters,
    apply_tsne,
    save_tsne_plots,
    save_cluster_samples
)
# Import GUI components (optional import to avoid tkinter dependency when not needed)
try:
    from synapse.gui import SynapseGUI
except ImportError:
    # GUI components may not be available if tkinter is not installed
    pass

# Export the most commonly used components
__all__ = [
    'Vgg3D', 
    'load_model_from_checkpoint',
    'SynapseDataset',
    'SynapseDataset2',
    'Synapse3DProcessor',
    'SynapseDataLoader',
    'config',
    'create_gif_from_volume',
    'visualize_specific_sample',
    'visualize_all_samples_from_bboxes',
    # Clustering exports
    'load_and_cluster_features',
    'find_random_samples_in_clusters',
    'find_closest_samples_in_clusters',
    'apply_tsne',
    'save_tsne_plots',
    'save_cluster_samples',
    # GUI exports (will be None if tkinter is not available)
    'SynapseGUI'
] 