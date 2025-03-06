import argparse
import os

class SynapseConfig:
    def __init__(self):
        # Default configuration values
        self.raw_base_dir = 'data/7_bboxes_plus_seg/raw'
        self.seg_base_dir = 'data/7_bboxes_plus_seg/seg'
        self.add_mask_base_dir = 'data/vesicle_cloud__syn_interface__mitochondria_annotation'
        self.bbox_name = ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']
        self.excel_file = 'data/7_bboxes_plus_seg'
        self.csv_output_dir = 'results/csv_outputs'
        self.size = (80, 80)
        self.subvol_size = 80
        self.num_frames = 80
        self.save_gifs_dir = 'results/gifs'
        self.alpha = 1.0
        self.segmentation_type = 10
        
        # CSV file configurations
        self.clustering_output_dir = 'results/clustering_results_final'
        
    def parse_args(self):
        """Parse command line arguments and update config values"""
        parser = argparse.ArgumentParser(description="Synapse Dataset Configuration")
        parser.add_argument('--raw_base_dir', type=str, default=self.raw_base_dir)
        parser.add_argument('--seg_base_dir', type=str, default=self.seg_base_dir)
        parser.add_argument('--add_mask_base_dir', type=str, default=self.add_mask_base_dir)
        parser.add_argument('--bbox_name', type=str, default=self.bbox_name, nargs='+')
        parser.add_argument('--excel_file', type=str, default=self.excel_file)
        parser.add_argument('--csv_output_dir', type=str, default=self.csv_output_dir)
        parser.add_argument('--size', type=tuple, default=self.size)
        parser.add_argument('--subvol_size', type=int, default=self.subvol_size)
        parser.add_argument('--num_frames', type=int, default=self.num_frames)
        parser.add_argument('--save_gifs_dir', type=str, default=self.save_gifs_dir)
        parser.add_argument('--alpha', type=float, default=self.alpha)
        parser.add_argument('--segmentation_type', type=int, default=self.segmentation_type, 
                           choices=range(0, 13), help='Type of segmentation overlay')
        
        args, _ = parser.parse_known_args()
        
        # Update config with parsed arguments
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        return self
    
    def get_feature_paths(self, segmentation_types=None, alphas=None):
        """Get paths to feature CSV files based on segmentation types and alphas"""
        if segmentation_types is None:
            segmentation_types = [9, 10]  # Default segmentation types
        
        if alphas is None:
            alphas = [1.0]  # Default alpha values
        
        paths = []
        for seg_type in segmentation_types:
            for alpha in alphas:
                alpha_str = str(alpha).replace('.', '_')
                filename = f'features_seg{seg_type}_alpha{alpha_str}.csv'
                paths.append(os.path.join(self.csv_output_dir, filename))
        
        return paths

# Create a singleton instance that can be imported
config = SynapseConfig() 