import os
import argparse
import yaml
from pathlib import Path

class ArgumentParser:
    """
    A class to handle command-line argument parsing for the SynapseClusterEM project.
    Encapsulates argument definition, parsing, and configuration file loading.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the ArgumentParser with an optional logger.
        
        Args:
            logger: Logger instance for logging messages
        """
        self.logger = logger
        
    def parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Synapse cluster analysis from 3D EM data',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Mode selection
        parser.add_argument('--mode', type=str, default='all',
                            choices=['preprocess', 'extract', 'cluster', 'visualize', 'all'],
                            help='Mode of operation')
        
        # Configuration file
        parser.add_argument('--config', type=str, default='config.yaml',
                            help='Path to a YAML configuration file (defaults to config.yaml)')
        
        # Data paths
        parser.add_argument('--raw_base_dir', type=str, default='data/raw',
                            help='Base directory for raw data')
        parser.add_argument('--seg_base_dir', type=str, default='data/seg',
                            help='Base directory for segmentation data')
        parser.add_argument('--add_mask_base_dir', type=str, default=None,
                            help='Base directory for additional mask data')
        parser.add_argument('--excel_dir', type=str, default='data',
                            help='Directory containing Excel files with synapse data')
        parser.add_argument('--output_dir', type=str, default='outputs',
                            help='Output directory for results')
        parser.add_argument('--checkpoint_path', type=str, required=False,
                            default='hemibrain_production.checkpoint',
                            help='Path to the model checkpoint')
        
        # Parameters
        parser.add_argument('--bbox_names', nargs='+', default=['bbox1', 'bbox2'],
                            help='Names of bounding boxes to process')
        parser.add_argument('--size', nargs=2, type=int, default=[80, 80],
                            help='Size of each frame (height, width)')
        parser.add_argument('--subvol_size', type=int, default=80,
                            help='Size of the cubic subvolume to extract')
        parser.add_argument('--num_frames', type=int, default=80,
                            help='Number of frames to use')
        parser.add_argument('--segmentation_types', nargs='+', type=int, default=[9, 10],
                            help='Segmentation types to process')
        parser.add_argument('--alphas', nargs='+', type=float, default=[1.0],
                            help='Alpha values for feature extraction')
        parser.add_argument('--n_clusters', type=int, default=10,
                            help='Number of clusters for clustering')
        parser.add_argument('--clustering_method', type=str, default='kmeans',
                            choices=['kmeans', 'hierarchical', 'dbscan'],
                            help='Clustering method to use')
        parser.add_argument('--batch_size', type=int, default=2,
                            help='Batch size for feature extraction')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='Number of worker processes for data loading')
        parser.add_argument('--create_3d_plots', action='store_true',
                            help='Create 3D plots of the UMAP embeddings')
        parser.add_argument('--save_interactive', action='store_true',
                            help='Save interactive plots')
        parser.add_argument('--gpu_id', type=int, default=0,
                            help='GPU ID to use (-1 for CPU)')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
        parser.add_argument('--force_recompute', action='store_true',
                            help='Force recomputation of features even if they already exist')
        parser.add_argument('--verbose', action='store_true',
                            help='Enable verbose output')
        parser.add_argument('--device', type=str, default='cuda:0',
                            help='Device to use for computation (e.g., "cpu", "cuda:0")')
        
        args = parser.parse_args()
        
        # Load configuration from file if provided
        if args.config:
            args = self._load_config_file(args)
            
        return args
    
    def _load_config_file(self, args):
        """
        Load configuration from YAML file and update args.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Updated args with values from config file
        """
        if os.path.exists(args.config):
            try:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                    # Update args with config values (only for keys that exist in args)
                    arg_dict = vars(args)
                    for key, value in config.items():
                        if key in arg_dict:
                            arg_dict[key] = value
                if self.logger:
                    self.logger.info(f"Loaded configuration from {args.config}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error loading configuration from {args.config}: {str(e)}")
        else:
            if self.logger:
                self.logger.warning(f"Configuration file {args.config} not found. Using default values.")
        
        return args
    
    def config_to_args(self, config):
        """
        Convert configuration dictionary to argparse namespace.
        
        Args:
            config: Configuration dictionary loaded from YAML
            
        Returns:
            argparse.Namespace object with configuration values
        """
        args = argparse.Namespace()
        
        # Workflow control
        args.mode = config.get('mode', 'all')
        
        # Data paths
        data_paths = config.get('data_paths', {})
        args.raw_base_dir = data_paths.get('raw_base_dir')
        args.seg_base_dir = data_paths.get('seg_base_dir')
        args.add_mask_base_dir = data_paths.get('add_mask_base_dir', '')
        args.excel_dir = data_paths.get('excel_dir')
        args.output_dir = data_paths.get('output_dir', 'outputs/default')
        args.checkpoint_path = data_paths.get('checkpoint_path')
        
        # Dataset parameters
        dataset = config.get('dataset', {})
        args.bbox_names = dataset.get('bbox_names', ['bbox1'])
        args.size = dataset.get('size', [80, 80])
        args.subvol_size = dataset.get('subvol_size', 80)
        args.num_frames = dataset.get('num_frames', 80)
        
        # Analysis parameters
        analysis = config.get('analysis', {})
        args.segmentation_types = analysis.get('segmentation_types', [9, 10])
        args.alphas = analysis.get('alphas', [1.0])
        args.n_clusters = analysis.get('n_clusters', 10)
        args.clustering_method = analysis.get('clustering_method', 'kmeans')
        args.batch_size = analysis.get('batch_size', 2)
        args.num_workers = analysis.get('num_workers', 0)
        
        # Visualization parameters
        visualization = config.get('visualization', {})
        args.create_3d_plots = visualization.get('create_3d_plots', False)
        args.save_interactive = visualization.get('save_interactive', False)
        
        # System parameters
        system = config.get('system', {})
        args.gpu_id = system.get('gpu_id', 0)
        args.seed = system.get('seed', 42)
        args.verbose = system.get('verbose', True)
        
        return args
    
    def load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if self.logger:
                self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return None 