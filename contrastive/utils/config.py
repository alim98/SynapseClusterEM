import argparse
import os
import sys
import json

# Add the parent directory to the path so we can import from synapse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the original config
from synapse.utils.config import SynapseConfig

class ContrastiveConfig(SynapseConfig):
    """Configuration for contrastive learning."""
    
    def __init__(self):
        # Initialize the parent class
        super().__init__()
        
        # Training parameters
        self.epochs = 100
        self.warmup_epochs = 10  # Number of epochs for warmup phase
        self.gradual_epochs = 20  # Number of epochs for gradual unfreezing
        self.batch_size = 32
        self.gradient_accumulation_steps = 16  # Added gradient accumulation
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.temperature = 0.07
        
        # Model parameters
        self.projection_dim = 128
        self.feature_dim = 512  # VGG's feature dimension
        
        # Augmentation parameters
        self.augmentation_prob = 0.5
        self.rotation_range = 15  # Maximum rotation angle in degrees
        self.flip_prob = 0.5  # Probability of random flip
        self.noise_level = 0.05  # Maximum noise level for Gaussian noise
        self.contrast_range = 0.2  # Range for random contrast adjustment
        self.brightness_range = 0.2  # Range for random brightness adjustment
        self.gaussian_blur_sigma = 1.0  # Sigma for Gaussian blur
        self.scale_range = (0.8, 1.2)  # Range for random scaling
        self.noise_std = 0.1  # Standard deviation for noise
        
        # Logging and checkpointing
        self.log_interval = 10
        self.save_interval = 5
        self.checkpoint_dir = "contrastive/checkpoints"
        self.log_dir = "contrastive/logs"
        
        # Feature comparison
        self.feature_comparison_interval = 10  # Compare features every N batches
        self.min_cosine_similarity = 0.8  # Minimum acceptable cosine similarity
        self.max_l2_distance = 0.1  # Maximum acceptable L2 distance
        
        # Memory optimization
        self.use_mixed_precision = True
        self.gradient_checkpointing = True
        self.memory_efficient_attention = True
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # New parameters
        self.num_epochs = 50
        self.results_dir = 'contrastive/results'
        self.visualization_dir = 'contrastive/visualization'

    def parse_args(self):
        # First, let the parent class parse its arguments
        super().parse_args()
        
        # Create a new parser for contrastive learning specific arguments
        parser = argparse.ArgumentParser(description="Contrastive Learning Configuration")
        
        # Training parameters
        parser.add_argument('--epochs', type=int, default=self.epochs)
        parser.add_argument('--warmup_epochs', type=int, default=self.warmup_epochs)
        parser.add_argument('--gradual_epochs', type=int, default=self.gradual_epochs)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--gradient_accumulation_steps', type=int, default=self.gradient_accumulation_steps)
        parser.add_argument('--learning_rate', type=float, default=self.learning_rate)
        parser.add_argument('--weight_decay', type=float, default=self.weight_decay)
        parser.add_argument('--temperature', type=float, default=self.temperature)
        
        # Model parameters
        parser.add_argument('--projection_dim', type=int, default=self.projection_dim)
        parser.add_argument('--feature_dim', type=int, default=self.feature_dim)
        
        # Augmentation parameters
        parser.add_argument('--augmentation_prob', type=float, default=self.augmentation_prob)
        parser.add_argument('--rotation_range', type=int, default=self.rotation_range)
        parser.add_argument('--flip_prob', type=float, default=self.flip_prob)
        parser.add_argument('--noise_level', type=float, default=self.noise_level)
        parser.add_argument('--contrast_range', type=float, default=self.contrast_range)
        parser.add_argument('--brightness_range', type=float, default=self.brightness_range)
        parser.add_argument('--gaussian_blur_sigma', type=float, default=self.gaussian_blur_sigma)
        parser.add_argument('--scale_range', type=str, default=str(self.scale_range))
        parser.add_argument('--noise_std', type=float, default=self.noise_std)
        
        # Directory parameters
        parser.add_argument('--checkpoint_dir', type=str, default=self.checkpoint_dir)
        parser.add_argument('--log_dir', type=str, default=self.log_dir)
        
        # Feature comparison parameters
        parser.add_argument('--feature_comparison_interval', type=int, default=self.feature_comparison_interval)
        parser.add_argument('--min_cosine_similarity', type=float, default=self.min_cosine_similarity)
        parser.add_argument('--max_l2_distance', type=float, default=self.max_l2_distance)
        
        # Memory optimization parameters
        parser.add_argument('--use_mixed_precision', type=bool, default=self.use_mixed_precision)
        parser.add_argument('--gradient_checkpointing', type=bool, default=self.gradient_checkpointing)
        parser.add_argument('--memory_efficient_attention', type=bool, default=self.memory_efficient_attention)
        
        # New parameters
        parser.add_argument('--num_epochs', type=int, default=self.num_epochs)
        parser.add_argument('--results_dir', type=str, default=self.results_dir)
        parser.add_argument('--visualization_dir', type=str, default=self.visualization_dir)
        
        args, _ = parser.parse_known_args()
        
        # Update the config with the parsed arguments
        for key, value in vars(args).items():
            if key in ['rotation_range', 'scale_range']:
                setattr(self, key, tuple(map(float, value.strip('()').split(','))))
            else:
                setattr(self, key, value)
        
        return self

    def to_dict(self):
        """Convert the config to a dictionary."""
        config_dict = super().to_dict()
        config_dict.update({
            'epochs': self.epochs,
            'warmup_epochs': self.warmup_epochs,
            'gradual_epochs': self.gradual_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'temperature': self.temperature,
            'projection_dim': self.projection_dim,
            'feature_dim': self.feature_dim,
            'augmentation_prob': self.augmentation_prob,
            'rotation_range': self.rotation_range,
            'flip_prob': self.flip_prob,
            'noise_level': self.noise_level,
            'contrast_range': self.contrast_range,
            'brightness_range': self.brightness_range,
            'gaussian_blur_sigma': self.gaussian_blur_sigma,
            'scale_range': self.scale_range,
            'noise_std': self.noise_std,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'feature_comparison_interval': self.feature_comparison_interval,
            'min_cosine_similarity': self.min_cosine_similarity,
            'max_l2_distance': self.max_l2_distance,
            'use_mixed_precision': self.use_mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing,
            'memory_efficient_attention': self.memory_efficient_attention,
            'num_epochs': self.num_epochs,
            'results_dir': self.results_dir,
            'visualization_dir': self.visualization_dir
        })
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def save(self, path):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path):
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Create a global instance of the config
config = ContrastiveConfig() 