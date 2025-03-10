import argparse
import os

class SynapseConfig:
    def __init__(self):
        self.raw_base_dir = 'data3/7_bboxes_plus_seg/raw'
        self.seg_base_dir = 'data3/7_bboxes_plus_seg/seg'
        self.add_mask_base_dir = 'data3/vesicle_cloud__syn_interface__mitochondria_annotation'
        self.bbox_name = ['bbox1' ]
        # self.bbox_name = ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']
        self.excel_file = 'data3/7_bboxes_plus_seg'
        self.csv_output_dir = 'results/csv_outputs'
        self.size = (80, 80)
        self.subvol_size = 80
        self.num_frames = 80
        self.save_gifs_dir = 'results/gifs'
        self.alpha = 1.0
        self.segmentation_type = 10
        
        self.gray_color = 0.6
        
        self.clustering_output_dir = 'results/clustering_results_final'
        self.report_output_dir = 'results/comprehensive_reports'
        
        # Clustering parameters
        self.clustering_algorithm = 'KMeans'  # Default clustering algorithm
        self.n_clusters = 2  # Default number of clusters for KMeans
        self.dbscan_eps = 0.5  # Default epsilon parameter for DBSCAN
        self.dbscan_min_samples = 5  # Default min_samples parameter for DBSCAN
        
    def parse_args(self):
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
        parser.add_argument('--gray_color', type=float, default=self.gray_color,
                           help='Gray color value (0-1) for overlaying segmentation')
        parser.add_argument('--clustering_output_dir', type=str, default=self.clustering_output_dir)
        parser.add_argument('--report_output_dir', type=str, default=self.report_output_dir)
        
        # Clustering parameters
        parser.add_argument('--clustering_algorithm', type=str, default=self.clustering_algorithm,
                           choices=['KMeans', 'DBSCAN'], help='Clustering algorithm to use')
        parser.add_argument('--n_clusters', type=int, default=self.n_clusters,
                           help='Number of clusters for KMeans')
        parser.add_argument('--dbscan_eps', type=float, default=self.dbscan_eps,
                           help='Epsilon parameter for DBSCAN')
        parser.add_argument('--dbscan_min_samples', type=int, default=self.dbscan_min_samples,
                           help='Minimum samples parameter for DBSCAN')
        
        args, _ = parser.parse_known_args()
        
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        return self
    
    def get_feature_paths(self, segmentation_types=None, alphas=None):
        if segmentation_types is None:
            segmentation_types = [9, 10]
        
        if alphas is None:
            alphas = [1.0]
        
        paths = []
        for seg_type in segmentation_types:
            for alpha in alphas:
                alpha_str = str(alpha).replace('.', '_')
                filename = f'features_seg{seg_type}_alpha{alpha_str}.csv'
                paths.append(os.path.join(self.csv_output_dir, filename))
        
        return paths

config = SynapseConfig()