import argparse
import os
import sys

# Add the parent directory to the path so we can import from synapse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the original config
from synapse.utils.config import SynapseConfig

class ContrastiveConfig(SynapseConfig):
    def __init__(self):
        # Initialize the parent class
        super().__init__()
        
        # Contrastive learning specific parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.temperature = 0.07  # Temperature parameter for NT-Xent loss
        self.proj_dim = 128      # Dimension of projection head output
        
        # Data augmentation parameters
        self.use_augmentation = True
        self.rotation_range = 15    # Degrees for 3D rotation
        self.flip_prob = 0.5        # Probability of random flip
        self.noise_level = 0.05     # Maximum noise level for Gaussian noise
        self.contrast_range = 0.2   # Range for random contrast adjustment
        self.brightness_range = 0.2 # Range for random brightness adjustment
        self.gaussian_blur_sigma = 1.0 # Sigma for Gaussian blur
        
        # Model checkpointing
        self.checkpoint_dir = 'contrastive/checkpoints'
        self.save_every = 5  # Save checkpoint every N epochs
        
        # Backbone model parameters
        self.backbone = 'vgg3d'  # The backbone architecture
        self.freeze_backbone = False  # Whether to freeze the backbone during training
        self.stage_to_finetune = -1  # Which stage to finetune (-1 means all)
        
        # Optimizer parameters
        self.optimizer = 'adam'  # 'adam' or 'sgd'
        self.momentum = 0.9     # Only used for SGD
        self.use_scheduler = True
        self.scheduler_type = 'cosine'  # 'cosine', 'step', or 'plateau'
        
        # Output directories
        self.results_dir = 'contrastive/results'
        self.log_dir = 'contrastive/logs'
        
        # Create output directories if they don't exist
        for directory in [self.checkpoint_dir, self.results_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

    def parse_args(self):
        # First, let the parent class parse its arguments
        super().parse_args()
        
        # Create a new parser for contrastive learning specific arguments
        parser = argparse.ArgumentParser(description="Contrastive Learning Configuration")
        
        # Training parameters
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--num_epochs', type=int, default=self.num_epochs)
        parser.add_argument('--learning_rate', type=float, default=self.learning_rate)
        parser.add_argument('--weight_decay', type=float, default=self.weight_decay)
        parser.add_argument('--temperature', type=float, default=self.temperature)
        parser.add_argument('--proj_dim', type=int, default=self.proj_dim)
        
        # Data augmentation parameters
        parser.add_argument('--use_augmentation', type=bool, default=self.use_augmentation)
        parser.add_argument('--rotation_range', type=float, default=self.rotation_range)
        parser.add_argument('--flip_prob', type=float, default=self.flip_prob)
        parser.add_argument('--noise_level', type=float, default=self.noise_level)
        parser.add_argument('--contrast_range', type=float, default=self.contrast_range)
        parser.add_argument('--brightness_range', type=float, default=self.brightness_range)
        parser.add_argument('--gaussian_blur_sigma', type=float, default=self.gaussian_blur_sigma)
        
        # Model parameters
        parser.add_argument('--backbone', type=str, default=self.backbone)
        parser.add_argument('--freeze_backbone', type=bool, default=self.freeze_backbone)
        parser.add_argument('--stage_to_finetune', type=int, default=self.stage_to_finetune)
        
        # Optimizer parameters
        parser.add_argument('--optimizer', type=str, default=self.optimizer)
        parser.add_argument('--momentum', type=float, default=self.momentum)
        parser.add_argument('--use_scheduler', type=bool, default=self.use_scheduler)
        parser.add_argument('--scheduler_type', type=str, default=self.scheduler_type)
        
        # Directory parameters
        parser.add_argument('--checkpoint_dir', type=str, default=self.checkpoint_dir)
        parser.add_argument('--results_dir', type=str, default=self.results_dir)
        parser.add_argument('--log_dir', type=str, default=self.log_dir)
        
        args, _ = parser.parse_known_args()
        
        # Update attributes with command line arguments
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        return self

# Create a global instance of the config
config = ContrastiveConfig() 