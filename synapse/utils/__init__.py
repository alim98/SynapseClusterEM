"""
Utility functions and helpers.
"""

from synapse.utils.config import SynapseConfig

# Create a singleton config instance for easy access
config = SynapseConfig()

__all__ = [
    'config', 
    'SynapseConfig',
] 