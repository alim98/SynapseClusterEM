import argparse
import os

class SynapseConfig:
    def __init__(self):
        self.raw_base_dir = 'data/7_bboxes_plus_seg/raw'
        self.seg_base_dir = 'data/7_bboxes_plus_seg/seg'
        self.add_mask_base_dir = 'data/vesicle_cloud__syn_interface__mitochondria_annotation'
        self.bbox_name = ['bbox1','bbox2','bbox3','bbox4','bbox5','bbox6','bbox7']
        self.excel_file = 'data/7_bboxes_plus_seg'
        self.csv_output_dir = 'results/csv_outputs'
        self.size = (80, 80)
        self.subvol_size = 80
        self.num_frames = 80
        self.save_gifs_dir = 'results/gifs'
        self.alpha = 1.0
        self.segmentation_type = 11
        # self.segmentation_type = [12, 10,11]  
        self.gray_color = 0.6
        
        self.clustering_output_dir = 'results/clustering_results_final'
        self.report_output_dir = 'results/comprehensive_reports'
        
        # Clustering parameters
        self.clustering_algorithm = 'KMeans'  # Default clustering algorithm
        self.n_clusters = 10 # Default number of clusters for KMeans
        self.dbscan_eps = 0.5  # Default epsilon parameter for DBSCAN
        self.dbscan_min_samples = 5  # Default min_samples parameter for DBSCAN
        self.results_dir='results'
        self.model_path='hemibrain_production.checkpoint'
        # Segmentation Type: 10
        #   Alpha: 1
        #   Extraction Method: stage_specific
        #   Layer Number: 20
        #   Normalize Volume: False
        #   Normalize Across Volume: True
        #   Smart Crop: True
        #   Presynapse Weight: 0.7
        #   Normalize Presynapse Size: True
        #   Target Percentage: None
        #   Size Tolerance: 0.1
        #   Absolute Difference: 0.00000578
        # Feature extraction parameters
        self.extraction_method = "stage_specific"  # Options: "standard" or "stage_specific"
        self.layer_num = 20  # Layer to extract features from when using stage_specific method
        self.preprocessing = 'intelligent_cropping'  # Options: 'normal' or 'intelligent_cropping'
        self.preprocessing_weights = 0.7 # it has opitons like 0.3 0.5 and 0.7
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
        
        # Feature extraction parameters
        parser.add_argument('--extraction_method', type=str, default=self.extraction_method,
                           choices=['standard', 'stage_specific'], 
                           help='Method to extract features ("standard" or "stage_specific")')
        parser.add_argument('--layer_num', type=int, default=self.layer_num,
                           help='Layer number to extract features from when using stage_specific method')
        
        args, _ = parser.parse_known_args()
        
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        return self
    
    def get_feature_paths(self, segmentation_types=None, alphas=None, extraction_method=None, layer_num=None):
        """
        Get paths to feature CSV files based on segmentation types, alphas, and extraction method.
        
        Args:
            segmentation_types: List of segmentation types to include (defaults to [9, 10])
            alphas: List of alpha values to include (defaults to [1.0])
            extraction_method: Feature extraction method ("standard" or "stage_specific")
            layer_num: Layer number for stage-specific extraction method
            
        Returns:
            list: List of file paths to feature CSV files
        """
        if segmentation_types is None:
            segmentation_types = [9, 10]
        
        if alphas is None:
            alphas = [1.0]
            
        if extraction_method is None:
            extraction_method = self.extraction_method
            
        if layer_num is None:
            layer_num = self.layer_num
        
        paths = []
        for seg_type in segmentation_types:
            for alpha in alphas:
                alpha_str = str(alpha).replace('.', '_')
                
                if extraction_method == 'stage_specific':
                    filename = f'features_layer{layer_num}_seg{seg_type}_alpha{alpha_str}.csv'
                else:
                    filename = f'features_seg{seg_type}_alpha{alpha_str}.csv'
                    
                paths.append(os.path.join(self.csv_output_dir, filename))
        
        return paths

config = SynapseConfig()