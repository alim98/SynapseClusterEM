#!/usr/bin/env python
"""
Comprehensive Report Generator for Synapse Analysis

This script analyzes the output directories created by inference.py and generates
a consolidated HTML report for all segmentation types and alpha combinations.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ReportGenerator')

class SynapseReportGenerator:
    """
    Generate comprehensive reports from synapse analysis outputs.
    Finds and summarizes all segmentation type and alpha combinations.
    """
    
    def __init__(self, 
                 csv_output_dir='results/csv_outputs', 
                 clustering_output_dir='results/clustering_results_final',
                 report_output_dir='results/comprehensive_reports'):
        """
        Initialize the report generator.
        
        Args:
            csv_output_dir: Directory containing feature CSVs and analysis results
            clustering_output_dir: Directory containing clustering results
            report_output_dir: Directory to save the generated reports
        """
        self.csv_output_dir = Path(csv_output_dir)
        self.clustering_output_dir = Path(clustering_output_dir)
        self.report_output_dir = Path(report_output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report directory
        os.makedirs(self.report_output_dir, exist_ok=True)
        
        # Initialize results storage
        self.seg_alpha_combinations = []
        self.feature_files = {}
        self.cluster_files = {}
        self.presynapse_files = {}
        self.visualizations = {}
        
        # CSS styles for the report
        self.css_styles = """
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            h1, h2, h3, h4 {
                color: #2c3e50;
                margin-top: 1.5em;
            }
            
            h1 {
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            
            .summary-box {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
            }
            
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            
            th {
                background-color: #f2f2f2;
            }
            
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            
            .nav-tabs {
                list-style: none;
                padding: 0;
                margin: 20px 0 0 0;
                border-bottom: 1px solid #ddd;
                display: flex;
                flex-wrap: wrap;
            }
            
            .nav-tabs li {
                margin-bottom: -1px;
            }
            
            .nav-tabs a {
                display: block;
                padding: 10px 15px;
                text-decoration: none;
                color: #555;
                border: 1px solid transparent;
                border-radius: 4px 4px 0 0;
            }
            
            .nav-tabs a:hover {
                background-color: #f5f5f5;
                border-color: #ddd #ddd transparent;
            }
            
            .nav-tabs a.active {
                background-color: #fff;
                color: #3498db;
                border-color: #ddd #ddd #fff;
                font-weight: bold;
            }
            
            .tab-content {
                padding: 20px;
                border: 1px solid #ddd;
                border-top: none;
            }
            
            .gallery {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin: 20px 0;
            }
            
            .gallery-item {
                margin: 10px;
                max-width: 350px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #fff;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }
            
            .gallery-item img {
                max-width: 100%;
                height: auto;
                border-radius: 3px;
            }
            
            .gallery-caption {
                margin-top: 5px;
                font-size: 0.9em;
                color: #666;
                text-align: center;
            }
            
            /* New styles for enhanced UMAP visualizations */
            .visualization-large {
                width: 100%;
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: #fff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .visualization-large img {
                max-width: 100%;
                height: auto;
                margin: 0 auto;
                display: block;
                border-radius: 4px;
            }
            
            .visualization-large .caption {
                margin-top: 12px;
                font-size: 1em;
                color: #555;
                text-align: center;
                line-height: 1.5;
                padding: 0 20px;
            }
            
            .interactive-link {
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }
            
            .button {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            
            .button:hover {
                background-color: #2980b9;
            }
            
            /* Styled toggle for showing/hiding sections */
            .section-toggle {
                cursor: pointer;
                color: #3498db;
                margin-left: 10px;
                font-size: 0.8em;
            }
            
            /* Highlight focus sections */
            .umap-section {
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin: 30px 0;
            }
            
            /* Card layouts for key visualizations */
            .visualization-cards {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
                margin: 30px 0;
            }
            
            .visualization-card {
                flex: 1 1 calc(33.333% - 20px);
                min-width: 300px;
                max-width: 500px;
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin: 10px 0;
                overflow: hidden;
                transition: transform 0.3s, box-shadow 0.3s;
            }
            
            .visualization-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.15);
            }
            
            .visualization-card img {
                width: 100%;
                max-height: 300px;
                object-fit: contain;
            }
            
            .visualization-card .card-content {
                padding: 20px;
                width: 100%;
            }
            
            .visualization-card h4 {
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 2px solid #f1f1f1;
                padding-bottom: 10px;
            }
        </style>
        """
    
    def copy_image_to_assets(self, source_path, report_dir):
        """
        Copy an image to the assets directory of a report.
        
        Args:
            source_path: Path to the source image
            report_dir: Directory of the report
            
        Returns:
            str: Relative path to the copied image for use in HTML
        """
        # Create assets directory if it doesn't exist
        assets_dir = os.path.join(report_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Extract the filename from the source path
        source_path = Path(source_path)
        filename = source_path.name
        
        # Define the destination path
        dest_path = os.path.join(assets_dir, filename)
        
        # Copy the file
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied image: {source_path} -> {dest_path}")
        except Exception as e:
            logger.error(f"Error copying image {source_path} to {dest_path}: {e}")
        
        # Return the relative path for HTML
        return f"assets/{filename}"
    
    def copy_to_assets(self, source_path, report_dir):
        """
        Copy a file to the assets directory of a report.
        
        Args:
            source_path: Path to the source file
            report_dir: Directory of the report
            
        Returns:
            str: Relative path to the copied file for use in HTML
        """
        # Create assets directory if it doesn't exist
        assets_dir = os.path.join(report_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Extract the filename from the source path
        source_path = Path(source_path)
        filename = source_path.name
        
        # Define the destination path
        dest_path = os.path.join(assets_dir, filename)
        
        # Copy the file
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied file: {source_path} -> {dest_path}")
        except Exception as e:
            logger.error(f"Error copying file {source_path} to {dest_path}: {e}")
        
        # Return the relative path for HTML
        return f"assets/{filename}"
    
    def find_available_combinations(self):
        """Find all available segmentation type and alpha combinations in the results directories."""
        logger.info("Finding available segmentation type and alpha combinations...")
        
        # Pattern to match feature CSV files: features_seg{seg_type}_alpha{alpha}.csv
        feature_pattern = re.compile(r'features_seg(\d+)_alpha(\d+(?:_\d+)?).csv')
        
        # Find all feature CSV files
        feature_files = glob.glob(str(self.csv_output_dir / '*.csv'))
        
        for file_path in feature_files:
            file_name = os.path.basename(file_path)
            match = feature_pattern.match(file_name)
            
            if match:
                seg_type = int(match.group(1))
                alpha_str = match.group(2)
                alpha = float(alpha_str.replace('_', '.'))
                
                combo = (seg_type, alpha)
                if combo not in self.seg_alpha_combinations:
                    self.seg_alpha_combinations.append(combo)
                    self.feature_files[combo] = file_path
        
        # Find matching cluster directories
        for seg_type, alpha in self.seg_alpha_combinations:
            alpha_str = str(alpha).replace('.', '_')
            
            # Check for segmentation-specific cluster directory
            cluster_dir = self.clustering_output_dir / f"seg{seg_type}_alpha{alpha_str}"
            if cluster_dir.exists():
                self.cluster_files[(seg_type, alpha)] = cluster_dir
            
            # Check for presynapse analysis
            presynapse_dir = self.clustering_output_dir / "presynapse_analysis" / f"seg{seg_type}_alpha{alpha_str}"
            if presynapse_dir.exists():
                self.presynapse_files[(seg_type, alpha)] = presynapse_dir
        
        logger.info(f"Found {len(self.seg_alpha_combinations)} segmentation type and alpha combinations")
        for seg_type, alpha in self.seg_alpha_combinations:
            logger.info(f"  - Segmentation Type: {seg_type}, Alpha: {alpha}")
        
        return self.seg_alpha_combinations
    
    def analyze_feature_data(self, seg_type, alpha):
        """
        Analyze feature data for a specific segmentation type and alpha value.
        
        Args:
            seg_type: Segmentation type
            alpha: Alpha value
            
        Returns:
            dict: Analysis results
        """
        combo = (seg_type, alpha)
        if combo not in self.feature_files:
            logger.warning(f"No feature file found for seg_type={seg_type}, alpha={alpha}")
            return {}
        
        file_path = self.feature_files[combo]
        logger.info(f"Analyzing feature data from {file_path}")
        
        try:
            # Load feature data
            features_df = pd.read_csv(file_path)
            
            # Basic statistics
            num_samples = len(features_df)
            num_features = len([c for c in features_df.columns if c.startswith('feat_')])
            bboxes = features_df['bbox_name'].unique() if 'bbox_name' in features_df.columns else []
            num_bboxes = len(bboxes)
            
            # Cluster information
            has_clusters = False
            num_clusters = 0
            cluster_counts = {}
            
            # First try to find cluster information in the features file
            if 'cluster' in features_df.columns:
                has_clusters = True
                clusters = features_df['cluster'].unique()
                num_clusters = len(clusters)
                cluster_counts = features_df['cluster'].value_counts().to_dict()
            
            # If not in features file, check for clustered_features.csv in combined_analysis directory
            if not has_clusters:
                combined_clustered_file = self.csv_output_dir / "combined_analysis" / "clustered_features.csv"
                if combined_clustered_file.exists():
                    try:
                        combined_df = pd.read_csv(combined_clustered_file)
                        # Filter for current segmentation type and alpha
                        if 'seg_type' in combined_df.columns and 'alpha' in combined_df.columns:
                            filtered_df = combined_df[
                                (combined_df['seg_type'] == seg_type) & 
                                (combined_df['alpha'] == alpha)
                            ]
                            if len(filtered_df) > 0 and 'cluster' in filtered_df.columns:
                                has_clusters = True
                                clusters = filtered_df['cluster'].unique()
                                num_clusters = len(clusters)
                                cluster_counts = filtered_df['cluster'].value_counts().to_dict()
                    except Exception as e:
                        logger.error(f"Error reading combined clustered file: {e}")
            
            # Also check for cluster sample images
            alpha_str = str(alpha).replace('.', '_')
            seg_dir = self.csv_output_dir / f"seg{seg_type}_alpha{alpha_str}"
            if seg_dir.exists():
                cluster_samples = list(seg_dir.glob("cluster_*_samples.png"))
                if cluster_samples:
                    has_clusters = True
                    # Extract cluster numbers from filenames
                    cluster_numbers = set()
                    for sample_path in cluster_samples:
                        filename = os.path.basename(sample_path)
                        match = re.search(r'cluster_(\d+)_samples\.png', filename)
                        if match:
                            cluster_numbers.add(int(match.group(1)))
                    
                    if cluster_numbers and (num_clusters == 0 or len(cluster_numbers) > num_clusters):
                        num_clusters = len(cluster_numbers)
            
            # UMAP information
            # Check for various possible UMAP column names
            umap_column_pairs = [
                ('umap_1', 'umap_2'),
                ('umap_x', 'umap_y'),
                ('UMAP_1', 'UMAP_2')
            ]
            
            has_umap = False
            for umap_1_col, umap_2_col in umap_column_pairs:
                if umap_1_col in features_df.columns and umap_2_col in features_df.columns:
                    has_umap = True
                    break
                    
            # If not found in the features file, check for a tsne_2d visualization which indicates UMAP was performed
            if not has_umap:
                combined_dir = self.csv_output_dir / "combined_analysis"
                if combined_dir.exists():
                    tsne_2d = combined_dir / "tsne_2d_bbox.png"
                    if tsne_2d.exists():
                        has_umap = True
            
            # Presynapse information
            has_presynapse_id = False
            num_presynapses = 0
            multiple_synapse_presynapses = {}
            
            # Analyze presynapse information if available
            if 'presynapse_id' in features_df.columns:
                has_presynapse_id = True
                presynapse_ids = features_df['presynapse_id'].dropna().unique()
                num_presynapses = len(presynapse_ids[presynapse_ids > 0])
                
                if num_presynapses > 0:
                    presynapse_counts = features_df[features_df['presynapse_id'] > 0].groupby('presynapse_id').size()
                    multiple_synapse_presynapses = presynapse_counts[presynapse_counts > 1].to_dict()
            
            # Collect all results
            results = {
                'num_samples': num_samples,
                'num_features': num_features,
                'has_clusters': has_clusters,
                'num_clusters': num_clusters,
                'has_umap': has_umap,
                'has_presynapse_id': has_presynapse_id,
                'num_presynapses': num_presynapses,
                'num_bboxes': num_bboxes
            }
            
            # Add bbox counts if available
            if len(bboxes) > 0:
                bbox_counts = features_df['bbox_name'].value_counts().to_dict()
                results['bbox_counts'] = bbox_counts
            
            # Add cluster counts if available
            if has_clusters and cluster_counts:
                results['cluster_counts'] = cluster_counts
            
            # Add multiple synapse presynapses if available
            if multiple_synapse_presynapses:
                results['multiple_synapse_presynapses'] = multiple_synapse_presynapses
            
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing feature data: {e}")
            import traceback
            traceback.print_exc()
            return {
                'num_samples': 0,
                'num_features': 0,
                'has_clusters': False,
                'num_clusters': 0,
                'has_umap': False,
                'has_presynapse_id': False,
                'num_presynapses': 0,
                'num_bboxes': 0
            }

    def find_cluster_samples(self, seg_type, alpha):
        """
        Find cluster sample images for a specific segmentation type and alpha value.
        Performs a comprehensive search across multiple potential directories.
        
        Args:
            seg_type: Segmentation type
            alpha: Alpha value
            
        Returns:
            list: List of paths to cluster sample images
        """
        alpha_str = str(alpha).replace('.', '_')
        cluster_samples = []
        
        # Search patterns to try in different directories
        search_patterns = [
            "cluster_*_samples.png", 
            "*cluster*samples*.png",
            "*cluster*slice*.png"
        ]
        
        # Directories to search in
        search_dirs = [
            # Main CSV output directory for this segmentation/alpha
            self.csv_output_dir / f"seg{seg_type}_alpha{alpha_str}",
            
            # Clustering output directory
            self.clustering_output_dir / f"seg{seg_type}_alpha{alpha_str}",
            
            # Combined analysis directory
            self.csv_output_dir / "combined_analysis",
            
            # Root output directory
            self.csv_output_dir,
            
            # Root clustering directory
            self.clustering_output_dir
        ]
        
        # If presynapse analysis exists for this combination, add it to search dirs
        if (seg_type, alpha) in self.presynapse_files:
            presynapse_dir = self.presynapse_files[(seg_type, alpha)]
            search_dirs.append(presynapse_dir)
            
            # Also search in potential subdirectories
            for subdir in ["cluster_visualizations", "visualizations", "clustering", "analysis"]:
                search_dirs.append(presynapse_dir / subdir)
        
        # Search in all directories with all patterns
        for search_dir in search_dirs:
            if search_dir.exists() and search_dir.is_dir():
                for pattern in search_patterns:
                    found_samples = list(search_dir.glob(pattern))
                    if found_samples:
                        logger.info(f"Found {len(found_samples)} cluster samples in {search_dir} with pattern {pattern}")
                        cluster_samples.extend([str(path) for path in found_samples])
                
                # Also search one level deeper
                for subdir in search_dir.glob("*/"):
                    if subdir.is_dir():
                        for pattern in search_patterns:
                            found_samples = list(subdir.glob(pattern))
                            if found_samples:
                                logger.info(f"Found {len(found_samples)} cluster samples in {subdir} with pattern {pattern}")
                                cluster_samples.extend([str(path) for path in found_samples])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_samples = []
        for sample in cluster_samples:
            if sample not in seen:
                seen.add(sample)
                unique_samples.append(sample)
        
        logger.info(f"Found {len(unique_samples)} unique cluster sample images for seg_type={seg_type}, alpha={alpha}")
        return unique_samples

    def find_visualizations(self, seg_type, alpha):
        """
        Find visualizations related to a specific segmentation type and alpha value.
        
        Args:
            seg_type: Segmentation type
            alpha: Alpha value
            
        Returns:
            dict: Dictionary of visualization paths
        """
        alpha_str = str(alpha).replace('.', '_')
        
        # Common visualization paths
        visualizations = {}
        cluster_samples = []
        
        # Combined analysis visualizations
        combined_dir = self.csv_output_dir / "combined_analysis"
        if combined_dir.exists():
            # t-SNE visualizations
            tsne_2d_bbox = combined_dir / f"tsne_2d_bbox.png"
            if tsne_2d_bbox.exists():
                visualizations['tsne_2d_bbox'] = str(tsne_2d_bbox)
            
            tsne_2d_cluster = combined_dir / f"tsne_2d_cluster.png"
            if tsne_2d_cluster.exists():
                visualizations['tsne_2d_cluster'] = str(tsne_2d_cluster)
            
            tsne_3d = combined_dir / f"tsne_3d.png"
            if tsne_3d.exists():
                visualizations['tsne_3d'] = str(tsne_3d)
                
            # Look for cluster samples in combined_analysis too
            combined_cluster_samples = list(combined_dir.glob("cluster_*_samples.png"))
            if combined_cluster_samples:
                cluster_samples.extend([str(path) for path in combined_cluster_samples])
        
        # Segmentation-specific visualizations
        seg_dir = self.csv_output_dir / f"seg{seg_type}_alpha{alpha_str}"
        if seg_dir.exists():
            # Find cluster sample visualizations with multiple patterns
            for pattern in ["cluster_*_samples.png", "*cluster*samples*.png"]:
                found_samples = list(seg_dir.glob(pattern))
                if found_samples:
                    cluster_samples.extend([str(path) for path in found_samples])
        
        # Also check in clustering_output_dir
        cluster_output_dir = self.clustering_output_dir / f"seg{seg_type}_alpha{alpha_str}"
        if cluster_output_dir.exists():
            # Check for additional cluster visualizations with multiple patterns
            for pattern in ["cluster_*_samples.png", "*cluster*samples*.png"]:
                found_samples = list(cluster_output_dir.glob(pattern))
                if found_samples:
                    cluster_samples.extend([str(path) for path in found_samples])
        
        # Presynapse analysis visualizations
        if (seg_type, alpha) in self.presynapse_files:
            presynapse_dir = self.presynapse_files[(seg_type, alpha)]
            
            # Connected UMAP - try multiple potential locations
            for umap_path in [
                presynapse_dir / "cluster_visualizations" / "connected_umap_visualization.png",
                presynapse_dir / "connected_umap_visualization.png",
                presynapse_dir / "umap_visualization.png"
            ]:
                if umap_path.exists():
                    visualizations['connected_umap'] = str(umap_path)
                    break
            
            # UMAP colored by bounding box - try multiple potential locations
            for umap_bbox_path in [
                presynapse_dir / "cluster_visualizations" / "umap_bbox_colored.png",
                presynapse_dir / "umap_bbox_colored.png",
                presynapse_dir / "umap_bbox.png"
            ]:
                if umap_bbox_path.exists():
                    visualizations['umap_bbox_colored'] = str(umap_bbox_path)
                    break
            
            # UMAP colored by cluster - try multiple potential locations
            for umap_cluster_path in [
                presynapse_dir / "cluster_visualizations" / "umap_cluster_colored.png",
                presynapse_dir / "umap_cluster_colored.png",
                presynapse_dir / "umap_cluster.png"
            ]:
                if umap_cluster_path.exists():
                    visualizations['umap_cluster_colored'] = str(umap_cluster_path)
                    break
            
            # Interactive UMAP - try multiple potential locations
            for interactive_umap_path in [
                presynapse_dir / "cluster_visualizations" / "connected_umap_interactive.html",
                presynapse_dir / "connected_umap_interactive.html"
            ]:
                if interactive_umap_path.exists():
                    visualizations['interactive_umap'] = str(interactive_umap_path)
                    break
            
            # Distance comparison visualizations
            distance_comparison_dir = presynapse_dir / "distance_comparison"
            if distance_comparison_dir.exists():
                distance_comparison_files = list(distance_comparison_dir.glob("*.png"))
                if distance_comparison_files:
                    visualizations['distance_comparison'] = [str(path) for path in distance_comparison_files]
            
            # Presynapse analysis report
            presynapse_report = presynapse_dir / "presynapse_analysis_report.html"
            if presynapse_report.exists():
                visualizations['presynapse_report'] = str(presynapse_report)
            
            # Check for cluster samples in presynapse directory too with multiple patterns
            for pattern in ["cluster_*_samples.png", "*cluster*samples*.png"]:
                found_samples = list(presynapse_dir.glob(pattern))
                if found_samples:
                    cluster_samples.extend([str(path) for path in found_samples])
            
            # Also check in potential subdirectories of presynapse_dir
            for subdir_name in ["cluster_visualizations", "visualizations", "clustering"]:
                subdir = presynapse_dir / subdir_name
                if subdir.exists() and subdir.is_dir():
                    for pattern in ["cluster_*_samples.png", "*cluster*samples*.png"]:
                        found_samples = list(subdir.glob(pattern))
                        if found_samples:
                            logger.info(f"Found {len(found_samples)} cluster samples in {subdir}")
                            cluster_samples.extend([str(path) for path in found_samples])
            
            # Also check in all subdirectories
            for subdir in presynapse_dir.glob("**/"):
                if subdir.is_dir() and subdir != presynapse_dir:
                    for pattern in ["cluster_*_samples.png", "*cluster*samples*.png"]:
                        found_samples = list(subdir.glob(pattern))
                        if found_samples:
                            logger.info(f"Found {len(found_samples)} cluster samples in {subdir}")
                            cluster_samples.extend([str(path) for path in found_samples])
        
        # Remove duplicate cluster samples
        if cluster_samples:
            unique_samples = list(dict.fromkeys(cluster_samples))
            logger.info(f"Found {len(unique_samples)} unique cluster sample images")
            visualizations['cluster_samples'] = unique_samples
        
        # Log what we found
        logger.info(f"Found {len(visualizations)} visualization types for seg_type={seg_type}, alpha={alpha}")
        for viz_type, paths in visualizations.items():
            if isinstance(paths, list):
                logger.info(f"  - {viz_type}: {len(paths)} files")
            else:
                logger.info(f"  - {viz_type}")
        
        return visualizations

    def generate_combination_report(self, seg_type, alpha, report_dir):
        """
        Generate a report for a specific segmentation type and alpha value.
        
        Args:
            seg_type: Segmentation type
            alpha: Alpha value
            report_dir: Directory to save the report
            
        Returns:
            tuple: (report_filename, all_html_content)
        """
        # Analyze feature data for this combination
        results = self.analyze_feature_data(seg_type, alpha)
        
        # Get the segmentation directory
        alpha_str = str(alpha).replace('.', '_')
        seg_dir = self.csv_output_dir / f"seg{seg_type}_alpha{alpha_str}"
        
        # Find available visualizations
        if (seg_type, alpha) not in self.visualizations:
            self.visualizations[(seg_type, alpha)] = self.find_visualizations(seg_type, alpha)
        viz = self.visualizations[(seg_type, alpha)]
        
        # Generate HTML content
        title = f"Synapse Analysis Results - Segmentation Type {seg_type}, Alpha {alpha}"
        
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.css_styles}
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Synapses</td>
                        <td>{results['num_samples']}</td>
                    </tr>
                    <tr>
                        <td>Number of Features</td>
                        <td>{results['num_features']}</td>
                    </tr>
        '''
        
        # Add number of bounding boxes if available
        if 'bbox_counts' in results and results['bbox_counts']:
            html += f'''
                    <tr>
                        <td>Number of Bounding Boxes</td>
                        <td>{len(results['bbox_counts'])}</td>
                    </tr>
            '''
        
        # Add clustering information if available
        if results['has_clusters']:
            html += f'''
                    <tr>
                        <td>Number of Clusters</td>
                        <td>{results['num_clusters']}</td>
                    </tr>
            '''
        
        # Add UMAP information if available
        if results['has_umap']:
            html += f'''
                    <tr>
                        <td>UMAP Visualization</td>
                        <td>Available</td>
                    </tr>
            '''
        
        # Add presynapse information if available
        if results['has_presynapse_id']:
            html += f'''
                    <tr>
                        <td>Presynapse Analysis</td>
                        <td>Available</td>
                    </tr>
                    <tr>
                        <td>Number of Presynapses</td>
                        <td>{results['num_presynapses']}</td>
                    </tr>
                    <tr>
                        <td>Presynapses with Multiple Synapses</td>
                        <td>{len(results['multiple_synapse_presynapses'])}</td>
                    </tr>
            '''
        
        html += '''
                </table>
            </div>
        '''
        
        # Add bounding box distribution if available
        if 'bbox_counts' in results and results['bbox_counts']:
            html += '''
            <div class="summary-box">
                <h2>Bounding Box Distribution</h2>
                <table>
                    <tr>
                        <th>Bounding Box Name</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
            '''
            
            total_count = sum(results['bbox_counts'].values())
            for bbox, count in sorted(results['bbox_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_count * 100
                html += f'''
                <tr>
                    <td>{bbox}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                '''
            
            html += '''
                </table>
            </div>
            '''
        
        # Add cluster distribution if available
        if results['has_clusters'] and 'cluster_counts' in results and results['cluster_counts']:
            html += '''
            <div class="summary-box">
                <h2>Cluster Distribution</h2>
                <table>
                    <tr>
                        <th>Cluster ID</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
            '''
            
            total_count = sum(results['cluster_counts'].values())
            for cluster, count in sorted(results['cluster_counts'].items(), key=lambda x: int(x[0])):
                percentage = count / total_count * 100
                html += f'''
                <tr>
                    <td>{cluster}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                '''
            
            html += '''
                </table>
            </div>
            '''
        
        # Add presynapse distribution if available
        if results['has_presynapse_id'] and results['multiple_synapse_presynapses']:
            html += '''
            <div class="summary-box">
                <h2>Presynapses with Multiple Synapses</h2>
                <table>
                    <tr>
                        <th>Presynapse ID</th>
                        <th>Number of Synapses</th>
                    </tr>
            '''
            
            for pre_id, count in sorted(results['multiple_synapse_presynapses'].items(), key=lambda x: x[1], reverse=True):
                html += f'''
                <tr>
                    <td>{pre_id}</td>
                    <td>{count}</td>
                </tr>
                '''
            
            html += '''
                </table>
            </div>
            '''
        
        # Add t-SNE visualizations if available
        if 'tsne_2d_bbox' in viz or 'tsne_2d_cluster' in viz or 'tsne_3d' in viz:
            html += '''
            <div class="summary-box">
                <h2>t-SNE Visualizations</h2>
                <p>t-SNE (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that helps visualize high-dimensional data in 2D or 3D space.</p>
                <div class="gallery">
            '''
            
            if 'tsne_2d_bbox' in viz:
                asset_path = self.copy_image_to_assets(viz['tsne_2d_bbox'], report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="t-SNE 2D colored by bounding box">
                    <div class="gallery-caption">t-SNE 2D Colored by Bounding Box</div>
                </div>
                '''
            
            if 'tsne_2d_cluster' in viz:
                asset_path = self.copy_image_to_assets(viz['tsne_2d_cluster'], report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="t-SNE 2D colored by cluster">
                    <div class="gallery-caption">t-SNE 2D Colored by Cluster</div>
                </div>
                '''
            
            if 'tsne_3d' in viz:
                asset_path = self.copy_image_to_assets(viz['tsne_3d'], report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="t-SNE 3D">
                    <div class="gallery-caption">t-SNE 3D Visualization</div>
                </div>
                '''
            
            html += '''
                </div>
            </div>
            '''
        
        # Add UMAP visualizations if available
        if 'connected_umap' in viz or 'umap_bbox_colored' in viz or 'umap_cluster_colored' in viz:
            html += '''
            <div class="summary-box">
                <h2>UMAP Visualizations</h2>
            '''
            
            html += '<p>UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique similar to t-SNE that helps visualize high-dimensional data in 2D space.</p>'
            
            # Add a dedicated section for the bbox-colored UMAP if available
            if 'umap_bbox_colored' in viz:
                asset_path = self.copy_image_to_assets(viz['umap_bbox_colored'], report_dir)
                html += f'''
                <h4>UMAP Colored by Bounding Box</h4>
                <div class="visualization-large">
                    <img src="{asset_path}" alt="UMAP colored by bounding box">
                    <div class="caption">UMAP visualization with points colored by their source bounding box. This helps identify patterns and groupings based on the anatomical region.</div>
                </div>
                '''
            
            # Add a dedicated section for the cluster-colored UMAP if available
            if 'umap_cluster_colored' in viz:
                asset_path = self.copy_image_to_assets(viz['umap_cluster_colored'], report_dir)
                html += f'''
                <h4>UMAP Colored by Cluster</h4>
                <div class="visualization-large">
                    <img src="{asset_path}" alt="UMAP colored by cluster">
                    <div class="caption">UMAP visualization with points colored by their assigned cluster. This reveals the structural relationships between different synapse types.</div>
                </div>
                '''
            
            # Add the standard connected UMAP
            if 'connected_umap' in viz:
                asset_path = self.copy_image_to_assets(viz['connected_umap'], report_dir)
                html += f'''
                <h4>Connected UMAP Visualization</h4>
                <div class="visualization-large">
                    <img src="{asset_path}" alt="Connected UMAP">
                    <div class="caption">UMAP visualization with synapses from the same presynapse connected by lines. This shows how synapses from the same presynaptic neuron relate to each other.</div>
                </div>
                '''
            
            # Add link to interactive UMAP if available
            if 'interactive_umap' in viz:
                interactive_path = self.copy_to_assets(viz['interactive_umap'], report_dir)
                html += f'''
                <div class="interactive-link">
                    <a href="{interactive_path}" target="_blank" class="button">Open Interactive UMAP Visualization</a>
                    <p>Click to explore an interactive version with hover information and 3D rotation capabilities.</p>
                </div>
                '''
            
            html += '''
            </div>
            '''
                
        # Add cluster sample visualizations if available
        if 'cluster_samples' in viz and viz['cluster_samples']:
            html += '''
            <div class="summary-box">
                <h2>Cluster Analysis Slices</h2>
                <p>These visualizations show representative center slices from each identified cluster, helping to understand the morphological characteristics that define each group. Each cluster typically displays 4 representative synapse examples.</p>
                <div class="gallery">
            '''
            
            for viz_path in viz['cluster_samples']:
                viz_name = os.path.basename(viz_path).replace('.png', '')
                asset_path = self.copy_image_to_assets(viz_path, report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="{viz_name}">
                    <div class="gallery-caption">{viz_name.replace('_', ' ').title()}</div>
                </div>
                '''
            
            html += '''
                </div>
            </div>
            '''
        else:
            # Try harder to find cluster sample visualizations
            logger.info("No cluster samples found in visualizations dictionary. Trying direct search...")
            
            # Search patterns to try
            patterns = ["cluster_*_samples.png", "*cluster*samples*.png"]
            found_samples = []
            
            # Search in multiple directories
            search_dirs = [
                seg_dir,
                self.clustering_output_dir / f"seg{seg_type}_alpha{alpha_str}",
                self.csv_output_dir / "combined_analysis"
            ]
            
            if (seg_type, alpha) in self.presynapse_files:
                search_dirs.append(self.presynapse_files[(seg_type, alpha)])
            
            for search_dir in search_dirs:
                if search_dir.exists():
                    for pattern in patterns:
                        samples = list(search_dir.glob(pattern))
                        if samples:
                            found_samples.extend(samples)
                            logger.info(f"Found {len(samples)} samples in {search_dir} with pattern {pattern}")
            
            if found_samples:
                html += '''
                <div class="summary-box">
                    <h2>Cluster Analysis Slices</h2>
                    <p>These visualizations show representative center slices from each identified cluster, helping to understand the morphological characteristics that define each group. Each cluster typically displays 4 representative synapse examples.</p>
                    <div class="gallery">
                '''
                
                for sample_path in found_samples:
                    sample_name = os.path.basename(str(sample_path)).replace('.png', '')
                    asset_path = self.copy_image_to_assets(str(sample_path), report_dir)
                    html += f'''
                    <div class="gallery-item">
                        <img src="{asset_path}" alt="{sample_name}">
                        <div class="gallery-caption">{sample_name.replace('_', ' ').title()}</div>
                    </div>
                    '''
                
                html += '''
                    </div>
                </div>
                '''
        
        # Add distance comparison visualizations if available
        if 'distance_comparison' in viz and viz['distance_comparison']:
            html += '''
            <div class="summary-box">
                <h2>Distance Comparison Visualizations</h2>
                <div class="gallery">
            '''
            
            for viz_path in viz['distance_comparison']:
                viz_name = os.path.basename(viz_path).replace('distance_comparison_', '').replace('.png', '')
                asset_path = self.copy_image_to_assets(viz_path, report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="{viz_name}">
                    <div class="gallery-caption">{viz_name.replace('_', ' ').title()}</div>
                </div>
                '''
            
            html += '''
                </div>
            </div>
            '''
        
        # Add link to presynapse analysis report if available
        if 'presynapse_report' in viz:
            report_path = self.copy_to_assets(viz['presynapse_report'], report_dir)
            html += f'''
            <div class="summary-box">
                <h2>Presynapse Analysis</h2>
                <p>A detailed analysis of presynapses is available.</p>
                <div class="interactive-link">
                    <a href="{report_path}" target="_blank" class="button">Open Presynapse Analysis Report</a>
                </div>
            </div>
            '''
        
        html += '''
        </body>
        </html>
        '''
        
        # Write HTML content to file
        report_filename = f"report_seg{seg_type}_alpha{alpha_str}.html"
        report_path = os.path.join(report_dir, report_filename)
        with open(report_path, 'w') as f:
            f.write(html)
        
        return report_filename, html

    def generate_complete_report(self):
        """
        Generate a complete report for all segmentation types and alpha combinations.
        
        Returns:
            str: Path to the index.html file of the complete report
        """
        # Find all available combinations if not already done
        if not self.seg_alpha_combinations:
            self.find_available_combinations()
            
        if not self.seg_alpha_combinations:
            logger.warning("No segmentation type and alpha combinations found.")
            return None
            
        # Create a directory for the report
        report_dir = self.report_output_dir / f"report_{self.timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Create assets directory
        assets_dir = os.path.join(report_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Generate combination reports
        combination_reports = {}
        
        for seg_type, alpha in self.seg_alpha_combinations:
            try:
                filename, html = self.generate_combination_report(seg_type, alpha, report_dir)
                combination_reports[(seg_type, alpha)] = filename
                logger.info(f"Generated report for seg_type={seg_type}, alpha={alpha}")
            except Exception as e:
                logger.error(f"Error generating report for seg_type={seg_type}, alpha={alpha}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate index.html with links to individual reports
        index_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synapse Analysis Complete Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.css_styles}
        </head>
        <body>
            <h1>Synapse Analysis Complete Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <h2>Available Reports</h2>
                <p>Click on a segmentation type and alpha combination to view the detailed report.</p>
                
                <div class="tabs">
        """
        
        # Organize by segmentation type
        seg_types = sorted(set([st for st, _ in self.seg_alpha_combinations]))
        
        # Add tabs for each segmentation type
        tab_ids = []
        for seg_type in seg_types:
            tab_id = f"tab-seg{seg_type}"
            tab_ids.append(tab_id)
            
            index_html += f"""
                    <button class="tablink" onclick="openTab('{tab_id}')" id="{tab_id}-btn">Segmentation Type {seg_type}</button>
            """
        
        index_html += """
                </div>
        """
        
        # Add content for each tab
        for i, seg_type in enumerate(seg_types):
            tab_id = tab_ids[i]
            is_first = i == 0
            
            index_html += f"""
                <div id="{tab_id}" class="tabcontent" style="display: {'block' if is_first else 'none'}">
                    <h3>Segmentation Type {seg_type}</h3>
                    <table>
                        <tr>
                            <th>Alpha Value</th>
                            <th>Report Link</th>
                        </tr>
            """
            
            # Add rows for each alpha value for this segmentation type
            alphas = sorted([a for st, a in self.seg_alpha_combinations if st == seg_type])
            
            for alpha in alphas:
                if (seg_type, alpha) in combination_reports:
                    report_filename = combination_reports[(seg_type, alpha)]
                    index_html += f"""
                        <tr>
                            <td>{alpha}</td>
                            <td><a href="{report_filename}">View Report</a></td>
                        </tr>
                    """
            
            index_html += """
                    </table>
                </div>
            """
        
        # Add JavaScript for tab functionality
        index_html += """
            </div>
            
            <script>
            function openTab(tabId) {
                // Hide all tab content
                var tabcontent = document.getElementsByClassName("tabcontent");
                for (var i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                
                // Remove active class from all tablinks
                var tablinks = document.getElementsByClassName("tablink");
                for (var i = 0; i < tablinks.length; i++) {
                    tablinks[i].classList.remove("active");
                }
                
                // Show the specific tab and add active class to the button
                document.getElementById(tabId).style.display = "block";
                document.getElementById(tabId + "-btn").classList.add("active");
            }
            
            // Set the first tab as active by default
            document.getElementById("tab-seg""" + str(seg_types[0]) + """-btn").classList.add("active");
            </script>
            
        </body>
        </html>
        """
        
        # Write the index.html file
        index_path = os.path.join(report_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        logger.info(f"Complete report generated at {report_dir}")
        
        return index_path

    def generate_presynapse_summary(self):
        """
        Generate a summary report for all presynapse analysis results.
        
        Returns:
            str: Path to the generated summary report
        """
        # Find all available combinations if not already done
        if not self.seg_alpha_combinations:
            self.find_available_combinations()
            
        # Create a directory for the summary report
        summary_dir = self.report_output_dir / f"presynapse_summary_{self.timestamp}"
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create assets directory
        assets_dir = os.path.join(summary_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Load visualizations if not already loaded
        for seg_type, alpha in self.seg_alpha_combinations:
            if (seg_type, alpha) not in self.visualizations:
                self.visualizations[(seg_type, alpha)] = self.find_visualizations(seg_type, alpha)
        
        # Collect presynapse information
        presynapses_by_combo = {}
        
        # For each combination, look for presynapse analysis results
        for seg_type, alpha in self.seg_alpha_combinations:
            if (seg_type, alpha) in self.presynapse_files:
                # Get the directory for this combination
                presynapse_dir = self.presynapse_files[(seg_type, alpha)]
                
                # Convert alpha to string format for filenames
                alpha_str = str(alpha).replace('.', '_')
                
                # Find the updated features file that contains presynapse IDs
                updated_features_file = presynapse_dir / f"updated_features_seg{seg_type}_alpha{alpha_str}.csv"
                # If not found, try alternative naming pattern
                if not updated_features_file.exists():
                    updated_features_file = list(presynapse_dir.glob("updated_features*.csv"))
                    if not updated_features_file:
                        logger.warning(f"No updated features file found for seg_type={seg_type}, alpha={alpha}")
                        continue
                    updated_features_file = updated_features_file[0]
                
                try:
                    # Load the features file
                    features_df = pd.read_csv(updated_features_file)
                    
                    if 'presynapse_id' not in features_df.columns:
                        continue
                    
                    # Count presynapses with multiple synapses
                    presynapse_counts = features_df[features_df['presynapse_id'] > 0].groupby('presynapse_id').size()
                    multiple_synapse_counts = presynapse_counts[presynapse_counts > 1]
                    
                    # Get cluster information if available
                    pre_groups = {}
                    if 'cluster' in features_df.columns:
                        for pre_id, pre_df in features_df[features_df['presynapse_id'] > 0].groupby('presynapse_id'):
                            # Only include presynapses with multiple synapses
                            if len(pre_df) > 1:
                                clusters = pre_df['cluster'].value_counts()
                                dominant_cluster = clusters.idxmax() if not clusters.empty else None
                                
                                pre_groups[pre_id] = {
                                    'count': len(pre_df),
                                    'clusters': clusters.to_dict(),
                                    'dominant_cluster': dominant_cluster,
                                    'dominant_percentage': (clusters.max() / len(pre_df) * 100) if not clusters.empty else 0
                                }
                    
                    presynapses_by_combo[(seg_type, alpha)] = pre_groups
                    
                except Exception as e:
                    logger.error(f"Error analyzing presynapse data for seg_type={seg_type}, alpha={alpha}: {e}")
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Presynapse Analysis Summary</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.css_styles}
        </head>
        <body>
            <h1>Presynapse Analysis Summary</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <h2>Overview</h2>
                <p>This report summarizes the results of presynapse analysis across all segmentation and alpha combinations.</p>
                
                <table>
                    <tr>
                        <th>Segmentation Type</th>
                        <th>Alpha Value</th>
                        <th>Presynapses with Multiple Synapses</th>
                    </tr>
        """
        
        # Add a row for each combination
        for (seg_type, alpha), pre_groups in sorted(presynapses_by_combo.items()):
            html_content += f"""
                <tr>
                    <td>{seg_type}</td>
                    <td>{alpha}</td>
                    <td>{len(pre_groups)}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Add sections for each combination
        for (seg_type, alpha), pre_groups in sorted(presynapses_by_combo.items()):
            alpha_str = str(alpha).replace('.', '_')
            
            html_content += f"""
            <div class="summary-box">
                <h2>Segmentation Type {seg_type}, Alpha {alpha}</h2>
                
                <h3>Presynapses with Multiple Synapses</h3>
                <table>
                    <tr>
                        <th>Presynapse ID</th>
                        <th>Number of Synapses</th>
            """
            
            # Add cluster columns if available
            has_clusters = any('dominant_cluster' in info and info['dominant_cluster'] is not None for info in pre_groups.values())
            if has_clusters:
                html_content += """
                        <th>Dominant Cluster</th>
                        <th>Dominant Cluster %</th>
                """
            
            html_content += """
                    </tr>
            """
            
            # Add a row for each presynapse
            for pre_id, info in sorted(pre_groups.items()):
                html_content += f"""
                    <tr>
                        <td>{pre_id}</td>
                        <td>{info['count']}</td>
                """
                
                # Add cluster information if available
                if has_clusters:
                    dominant_cluster = info.get('dominant_cluster', 'N/A')
                    dominant_percentage = info.get('dominant_percentage', 0)
                    
                    html_content += f"""
                        <td>{dominant_cluster}</td>
                        <td>{dominant_percentage:.1f}%</td>
                    """
                
                html_content += """
                    </tr>
                """
            
            html_content += """
                </table>
                
                <!-- Include visualizations if available -->
            """
            
            # Get visualizations for this combo
            viz_dict = self.visualizations.get((seg_type, alpha), {})
            
            # Add UMAP visualizations with a focus on bbox and cluster coloring
            umap_visualizations = []
            
            # UMAP visualization by bounding box
            if 'umap_bbox_colored' in viz_dict:
                dest_path = self.copy_image_to_assets(viz_dict['umap_bbox_colored'], summary_dir)
                umap_visualizations.append({
                    'path': dest_path,
                    'title': 'UMAP Colored by Bounding Box',
                    'description': 'This visualization shows synapses colored by their source bounding box, helping identify patterns based on anatomical regions.'
                })
            
            # UMAP visualization by cluster
            if 'umap_cluster_colored' in viz_dict:
                dest_path = self.copy_image_to_assets(viz_dict['umap_cluster_colored'], summary_dir)
                umap_visualizations.append({
                    'path': dest_path,
                    'title': 'UMAP Colored by Cluster',
                    'description': 'This visualization shows synapses colored by their assigned cluster, revealing structural relationships between different synapse types.'
                })
            
            # Connected UMAP visualization
            if 'connected_umap' in viz_dict:
                dest_path = self.copy_image_to_assets(viz_dict['connected_umap'], summary_dir)
                umap_visualizations.append({
                    'path': dest_path,
                    'title': 'Connected UMAP Visualization',
                    'description': 'This visualization shows synapses from the same presynapse connected by lines.'
                })
            
            # Add all UMAP visualizations to the HTML in a card format
            if umap_visualizations:
                html_content += """
                <div class="umap-section">
                    <h3>UMAP Visualizations</h3>
                    <div class="visualization-cards">
                """
                
                for viz in umap_visualizations:
                    html_content += f"""
                        <div class="visualization-card">
                            <img src="{viz['path']}" alt="{viz['title']}">
                            <div class="card-content">
                                <h4>{viz['title']}</h4>
                                <p>{viz['description']}</p>
                            </div>
                        </div>
                    """
                
                html_content += """
                    </div>
                </div>
                """
                
                # Add link to interactive UMAP if available
                if 'interactive_umap' in viz_dict:
                    dest_path = self.copy_to_assets(viz_dict['interactive_umap'], summary_dir)
                    html_content += f"""
                    <div class="interactive-link">
                        <a href="{dest_path}" target="_blank" class="button">Open Interactive UMAP Visualization</a>
                        <p>Click to explore an interactive version with hover information.</p>
                    </div>
                    """
            else:
                # Try to manually find and add the connected UMAP visualization directly from the directory
                presynapse_dir = self.presynapse_files[(seg_type, alpha)]
                connected_umap = presynapse_dir / "cluster_visualizations" / "connected_umap_visualization.png"
                if connected_umap.exists():
                    logger.info(f"Found connected UMAP at {connected_umap}")
                    dest_path = self.copy_image_to_assets(str(connected_umap), summary_dir)
                    html_content += f"""
                    <h4>Connected UMAP Visualization</h4>
                    <img src="{dest_path}" alt="Connected UMAP" style="max-width: 100%;">
                    """
                else:
                    # Try alternative path patterns
                    logger.info("Looking for UMAP with alternative patterns")
                    potential_paths = [
                        presynapse_dir / "connected_umap_visualization.png",
                        presynapse_dir / "umap_visualization.png",
                        list(presynapse_dir.glob("*umap*.png")),
                    ]
                    
                    for path in potential_paths:
                        if isinstance(path, list) and path:
                            path = path[0]
                        if isinstance(path, Path) and path.exists():
                            logger.info(f"Found UMAP at {path}")
                            dest_path = self.copy_image_to_assets(str(path), summary_dir)
                            html_content += f"""
                            <h4>UMAP Visualization</h4>
                            <img src="{dest_path}" alt="UMAP" style="max-width: 100%;">
                            """
                            break
            
            # Add cluster sample visualizations if available
            if 'cluster_samples' in viz_dict and viz_dict['cluster_samples']:
                html_content += """
                <h3>Cluster Analysis Slices</h3>
                <p>Representative center slices from each identified cluster:</p>
                <div class="gallery">
                """
                
                for viz_path in viz_dict['cluster_samples']:
                    viz_name = os.path.basename(viz_path).replace('.png', '')
                    dest_path = self.copy_image_to_assets(viz_path, summary_dir)
                    html_content += f"""
                    <div class="gallery-item">
                        <img src="{dest_path}" alt="{viz_name}">
                        <div class="gallery-caption">{viz_name.replace('_', ' ').title()}</div>
                    </div>
                    """
                
                html_content += "</div>"
            
            html_content += """
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML content to file
        html_file = os.path.join(summary_dir, "presynapse_summary.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Presynapse summary report saved to {html_file}")
        
        return html_file