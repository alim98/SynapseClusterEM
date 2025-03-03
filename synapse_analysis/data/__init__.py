# Import key functions and classes for external use
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data,
    apply_global_normalization
)
from synapse_analysis.data.dataset import SynapseDataset

# Explicitly define what's exported
__all__ = [
    'Synapse3DProcessor',
    'load_all_volumes',
    'load_synapse_data',
    'apply_global_normalization',
    'SynapseDataset'
]
