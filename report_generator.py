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
                margin-right: 5px;
            }
            
            .nav-tabs a {
                display: block;
                padding: 10px 15px;
                text-decoration: none;
                color: #555;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 5px 5px 0 0;
            }
            
            .nav-tabs a.active {
                background-color: #fff;
                border-bottom: 1px solid #fff;
                margin-bottom: -1px;
                color: #2c3e50;
                font-weight: bold;
            }
            
            .tab-content {
                border: 1px solid #ddd;
                border-top: none;
                border-radius: 0 0 5px 5px;
                padding: 20px;
                margin-bottom: 20px;
            }
            
            .card {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 20px;
                overflow: hidden;
            }
            
            .card-header {
                background-color: #f8f9fa;
                padding: 15px;
                border-bottom: 1px solid #ddd;
            }
            
            .card-header h2 {
                margin: 0;
            }
            
            .gallery {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }
            
            .gallery-item {
                flex: 0 0 calc(33.333% - 20px);
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            
            .gallery-item img {
                width: 100%;
                height: auto;
                display: block;
                max-height: 300px;
                object-fit: contain;
            }
            
            .gallery-caption {
                padding: 10px;
                background-color: #f8f9fa;
                text-align: center;
                font-weight: bold;
            }
            
            @media (max-width: 768px) {
                .gallery-item {
                    flex: 0 0 calc(50% - 20px);
                }
            }
            
            @media (max-width: 480px) {
                .gallery-item {
                    flex: 0 0 100%;
                }
            }
        </style>
        """
    
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
            has_presynapse_id = 'presynapse_id' in features_df.columns
            num_presynapses = 0
            multiple_synapse_presynapses = {}
            
            if has_presynapse_id:
                # Count presynapses and identify those with multiple synapses
                presynapse_ids = features_df[features_df['presynapse_id'] > 0]['presynapse_id'].unique()
                num_presynapses = len(presynapse_ids)
                
                if num_presynapses > 0:
                    presynapse_counts = features_df[features_df['presynapse_id'] > 0].groupby('presynapse_id').size()
                    multiple_synapse_counts = presynapse_counts[presynapse_counts > 1]
                    multiple_synapse_presynapses = multiple_synapse_counts.to_dict()
            
            # If no presynapse_id in main file, check if there's a presynapse analysis directory
            if not has_presynapse_id:
                presynapse_dir = self.clustering_output_dir / "presynapse_analysis" / f"seg{seg_type}_alpha{alpha_str}"
                if presynapse_dir.exists():
                    # Look for updated features file with presynapse IDs
                    updated_features_file = presynapse_dir / f"updated_features_seg{seg_type}_alpha{alpha_str}.csv"
                    if updated_features_file.exists():
                        try:
                            presynapse_df = pd.read_csv(updated_features_file)
                            if 'presynapse_id' in presynapse_df.columns:
                                has_presynapse_id = True
                                presynapse_ids = presynapse_df[presynapse_df['presynapse_id'] > 0]['presynapse_id'].unique()
                                num_presynapses = len(presynapse_ids)
                                
                                if num_presynapses > 0:
                                    presynapse_counts = presynapse_df[presynapse_df['presynapse_id'] > 0].groupby('presynapse_id').size()
                                    multiple_synapse_counts = presynapse_counts[presynapse_counts > 1]
                                    multiple_synapse_presynapses = multiple_synapse_counts.to_dict()
                        except Exception as e:
                            logger.error(f"Error reading updated features file: {e}")
            
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
        
        # Segmentation-specific visualizations
        seg_dir = self.csv_output_dir / f"seg{seg_type}_alpha{alpha_str}"
        if seg_dir.exists():
            # Find cluster sample visualizations
            cluster_samples = list(seg_dir.glob("cluster_*_samples.png"))
            if cluster_samples:
                visualizations['cluster_samples'] = [str(path) for path in cluster_samples]
        
        # Presynapse analysis visualizations
        if (seg_type, alpha) in self.presynapse_files:
            presynapse_dir = self.presynapse_files[(seg_type, alpha)]
            
            # Connected UMAP
            connected_umap = presynapse_dir / "cluster_visualizations" / "connected_umap_visualization.png"
            if connected_umap.exists():
                visualizations['connected_umap'] = str(connected_umap)
            
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
        
        return visualizations
    
    def copy_image_to_assets(self, image_path, report_dir):
        """
        Copy an image file to the assets directory and return its relative path.
        
        Args:
            image_path: Original path to the image file
            report_dir: Report directory where assets are stored
            
        Returns:
            str: Relative path to the copied image file
        """
        if not image_path or not os.path.exists(image_path):
            return "#"  # Return a placeholder for missing images
        
        # Create assets directory if it doesn't exist
        assets_dir = os.path.join(report_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Create a filename for the copied image
        original_filename = os.path.basename(image_path)
        # Include a directory prefix to avoid name collisions
        parent_dir = os.path.basename(os.path.dirname(image_path))
        asset_filename = f"{parent_dir}_{original_filename}"
        asset_path = os.path.join(assets_dir, asset_filename)
        
        # Copy the file
        try:
            shutil.copy2(image_path, asset_path)
            logger.info(f"Copied {image_path} to {asset_path}")
            # Return relative path from the report HTML file to the asset
            return os.path.join("assets", asset_filename)
        except Exception as e:
            logger.error(f"Error copying image {image_path}: {e}")
            return "#"  # Return a placeholder for missing images

    def generate_combination_report(self, seg_type, alpha, report_dir):
        """
        Generate a report for a specific segmentation type and alpha value.
        
        Args:
            seg_type: Segmentation type
            alpha: Alpha value
            report_dir: Directory where the report is being generated
            
        Returns:
            str: HTML report content
        """
        combo = (seg_type, alpha)
        alpha_str = str(alpha).replace('.', '_')
        
        # Get feature data analysis
        feature_analysis = self.analyze_feature_data(seg_type, alpha)
        
        # Find visualizations
        viz = self.find_visualizations(seg_type, alpha)
        
        # Get segmentation type description
        seg_type_descriptions = {
            0: "Raw data",
            1: "Presynapse",
            2: "Postsynapse",
            3: "Both sides",
            4: "Vesicles + Cleft (closest only)",
            5: "(closest vesicles/cleft + sides)",
            6: "Vesicle cloud (closest)",
            7: "Cleft (closest)",
            8: "Mitochondria (closest)",
            9: "Vesicle + Cleft",
            10: "Cleft + Pre"
        }
        seg_description = seg_type_descriptions.get(seg_type, f"Unknown type ({seg_type})")
        
        # Create report HTML
        html = f"""
        <div class="card">
            <div class="card-header">
                <h2>Segmentation Type {seg_type} ({seg_description}) with Alpha {alpha}</h2>
            </div>
            
            <div class="summary-box">
                <h3>Summary</h3>
                <p>This report analyzes the results for segmentation type {seg_type} ({seg_description}) with alpha value {alpha}.</p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Number of Synapses</td>
                        <td>{feature_analysis.get('num_samples', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Number of Features</td>
                        <td>{feature_analysis.get('num_features', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Number of Bounding Boxes</td>
                        <td>{feature_analysis.get('num_bboxes', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Clustering Available</td>
                        <td>{"Yes" if feature_analysis.get('has_clusters', False) else "No"}</td>
                    </tr>
                    <tr>
                        <td>Number of Clusters</td>
                        <td>{feature_analysis.get('num_clusters', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>UMAP Available</td>
                        <td>{"Yes" if feature_analysis.get('has_umap', False) else "No"}</td>
                    </tr>
                    <tr>
                        <td>Presynapse Analysis Available</td>
                        <td>{"Yes" if feature_analysis.get('has_presynapse_id', False) else "No"}</td>
                    </tr>
                    <tr>
                        <td>Number of Presynapses</td>
                        <td>{feature_analysis.get('num_presynapses', 'N/A')}</td>
                    </tr>
                </table>
            </div>
        """
        
        # Add bbox distribution if available
        if 'bbox_counts' in feature_analysis:
            html += """
            <h3>Bounding Box Distribution</h3>
            <table>
                <tr>
                    <th>Bounding Box</th>
                    <th>Count</th>
                </tr>
            """
            
            for bbox, count in sorted(feature_analysis['bbox_counts'].items()):
                html += f"""
                <tr>
                    <td>{bbox}</td>
                    <td>{count}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add cluster distribution if available
        if 'cluster_counts' in feature_analysis:
            html += """
            <h3>Cluster Distribution</h3>
            <table>
                <tr>
                    <th>Cluster</th>
                    <th>Count</th>
                </tr>
            """
            
            for cluster, count in sorted(feature_analysis['cluster_counts'].items()):
                html += f"""
                <tr>
                    <td>{cluster}</td>
                    <td>{count}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add multiple-synapse presynapses if available
        if 'multiple_synapse_presynapses' in feature_analysis:
            html += """
            <h3>Presynapses with Multiple Synapses</h3>
            <table>
                <tr>
                    <th>Presynapse ID</th>
                    <th>Number of Synapses</th>
                </tr>
            """
            
            for pre_id, count in sorted(feature_analysis['multiple_synapse_presynapses'].items()):
                html += f"""
                <tr>
                    <td>{pre_id}</td>
                    <td>{count}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add visualizations
        html += "<h3>Visualizations</h3>"
        
        # Add t-SNE visualizations if available
        if 'tsne_2d_bbox' in viz or 'tsne_2d_cluster' in viz or 'tsne_3d' in viz:
            html += "<h4>t-SNE Visualizations</h4>"
            html += '<div class="gallery">'
            
            if 'tsne_2d_bbox' in viz:
                asset_path = self.copy_image_to_assets(viz['tsne_2d_bbox'], report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="t-SNE 2D by Bounding Box">
                    <div class="gallery-caption">t-SNE 2D colored by Bounding Box</div>
                </div>
                '''
            
            if 'tsne_2d_cluster' in viz:
                asset_path = self.copy_image_to_assets(viz['tsne_2d_cluster'], report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="t-SNE 2D by Cluster">
                    <div class="gallery-caption">t-SNE 2D colored by Cluster</div>
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
            
            html += '</div>'
        
        # Add connected UMAP if available
        if 'connected_umap' in viz:
            asset_path = self.copy_image_to_assets(viz['connected_umap'], report_dir)
            html += "<h4>Connected UMAP Visualization</h4>"
            html += f'''
            <div class="gallery-item">
                <img src="{asset_path}" alt="Connected UMAP">
                <div class="gallery-caption">UMAP visualization with synapses from the same presynapse connected</div>
            </div>
            '''
        
        # Add cluster sample visualizations if available
        if 'cluster_samples' in viz and viz['cluster_samples']:
            html += "<h4>Cluster Sample Visualizations</h4>"
            html += '<div class="gallery">'
            
            for viz_path in viz['cluster_samples']:
                viz_name = os.path.basename(viz_path).replace('.png', '')
                asset_path = self.copy_image_to_assets(viz_path, report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="{viz_name}">
                    <div class="gallery-caption">{viz_name.replace('_', ' ').title()}</div>
                </div>
                '''
            
            html += '</div>'
        
        # Add distance comparison visualizations if available
        if 'distance_comparison' in viz and viz['distance_comparison']:
            html += "<h4>Distance Comparison Visualizations</h4>"
            html += '<div class="gallery">'
            
            for viz_path in viz['distance_comparison']:
                viz_name = os.path.basename(viz_path).replace('distance_comparison_', '').replace('.png', '')
                asset_path = self.copy_image_to_assets(viz_path, report_dir)
                html += f'''
                <div class="gallery-item">
                    <img src="{asset_path}" alt="{viz_name}">
                    <div class="gallery-caption">{viz_name.replace('_', ' ').title()}</div>
                </div>
                '''
            
            html += '</div>'
        
        # Link to presynapse report if available
        if 'presynapse_report' in viz:
            # For HTML files, we'll copy them and provide a direct link
            asset_path = self.copy_image_to_assets(viz['presynapse_report'], report_dir).replace('.png', '.html')
            report_filename = os.path.basename(viz['presynapse_report'])
            
            # Copy the HTML file directly (not as an image)
            try:
                shutil.copy2(viz['presynapse_report'], os.path.join(report_dir, "assets", report_filename))
                asset_path = os.path.join("assets", report_filename)
            except Exception as e:
                logger.error(f"Error copying HTML report {viz['presynapse_report']}: {e}")
                asset_path = "#"
            
            html += f'''
            <h4>Presynapse Analysis</h4>
            <p>
                <a href="{asset_path}" target="_blank">
                    Detailed Presynapse Analysis Report
                </a>
            </p>
            '''
        
        # Close the card div
        html += "</div>"
        
        return html
    
    def generate_complete_report(self):
        """
        Generate a complete report for all segmentation type and alpha combinations.
        
        Returns:
            str: Path to the generated report
        """
        # Find all available combinations
        self.find_available_combinations()
        
        if not self.seg_alpha_combinations:
            logger.warning("No segmentation type and alpha combinations found")
            return None
        
        # Create a directory for this report
        report_dir = self.report_output_dir / f"report_{self.timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Create assets directory for copied images
        assets_dir = os.path.join(report_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Create the HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synapse Analysis Report - {self.timestamp}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.css_styles}
        </head>
        <body>
            <h1>Synapse Analysis Comprehensive Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <h2>Analysis Summary</h2>
                <p>This report summarizes the results of synapse analysis for {len(self.seg_alpha_combinations)} segmentation type and alpha combinations.</p>
                
                <table>
                    <tr>
                        <th>Segmentation Type</th>
                        <th>Alpha Value</th>
                        <th>Number of Synapses</th>
                        <th>Presynapse Analysis</th>
                    </tr>
        """
        
        # Add a row for each combination
        for seg_type, alpha in sorted(self.seg_alpha_combinations):
            feature_analysis = self.analyze_feature_data(seg_type, alpha)
            has_presynapse = (seg_type, alpha) in self.presynapse_files
            
            html += f"""
                <tr>
                    <td>{seg_type}</td>
                    <td>{alpha}</td>
                    <td>{feature_analysis.get('num_samples', 'N/A')}</td>
                    <td>{"Available" if has_presynapse else "Not Available"}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Add a tab for each combination
        html += """
            <div class="nav-tabs">
        """
        
        for i, (seg_type, alpha) in enumerate(sorted(self.seg_alpha_combinations)):
            active = ' class="active"' if i == 0 else ''
            html += f"""
                <li><a href="#tab-{seg_type}-{alpha}"{active}>Seg{seg_type}-Alpha{alpha}</a></li>
            """
        
        html += """
            </div>
            
            <div class="tab-content">
        """
        
        # Add content for each tab
        for i, (seg_type, alpha) in enumerate(sorted(self.seg_alpha_combinations)):
            style = '' if i == 0 else ' style="display: none;"'
            html += f"""
                <div id="tab-{seg_type}-{alpha}"{style}>
                    {self.generate_combination_report(seg_type, alpha, report_dir)}
                </div>
            """
        
        html += """
            </div>
            
            <script>
                // Simple tab navigation
                document.addEventListener('DOMContentLoaded', function() {
                    const tabs = document.querySelectorAll('.nav-tabs a');
                    const tabContents = document.querySelectorAll('.tab-content > div');
                    
                    tabs.forEach(tab => {
                        tab.addEventListener('click', function(e) {
                            e.preventDefault();
                            
                            // Deactivate all tabs
                            tabs.forEach(t => t.classList.remove('active'));
                            
                            // Hide all tab contents
                            tabContents.forEach(content => {
                                content.style.display = 'none';
                            });
                            
                            // Activate clicked tab
                            this.classList.add('active');
                            
                            // Show corresponding content
                            const tabId = this.getAttribute('href');
                            document.querySelector(tabId).style.display = 'block';
                        });
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Write the report to file
        report_path = report_dir / "index.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Report generated at {report_path}")
        return str(report_path)

    def generate_presynapse_summary(self):
        """
        Generate a summary report specifically focused on presynapse analysis across all segmentation types.
        
        Returns:
            str: Path to the generated report
        """
        # Find all available combinations
        self.find_available_combinations()
        
        if not self.seg_alpha_combinations:
            logger.warning("No segmentation type and alpha combinations found")
            return None
        
        # Create a directory for this report
        report_dir = self.report_output_dir / f"presynapse_summary_{self.timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Create assets directory for copied images
        assets_dir = os.path.join(report_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Collect presynapse data across all segmentation types
        presynapse_data = {}
        for seg_type, alpha in self.seg_alpha_combinations:
            combo = (seg_type, alpha)
            if combo in self.presynapse_files:
                # Find the updated features file with presynapse IDs
                presynapse_dir = self.presynapse_files[combo]
                alpha_str = str(alpha).replace('.', '_')
                features_file = presynapse_dir / f"updated_features_seg{seg_type}_alpha{alpha_str}.csv"
                
                if features_file.exists():
                    try:
                        df = pd.read_csv(features_file)
                        if 'presynapse_id' in df.columns:
                            # Count synapses per presynapse ID
                            presynapse_counts = df[df['presynapse_id'] > 0].groupby('presynapse_id').size()
                            multiple_synapse_presynapses = presynapse_counts[presynapse_counts > 1]
                            
                            presynapse_data[combo] = {
                                'total_presynapses': len(df[df['presynapse_id'] > 0]['presynapse_id'].unique()),
                                'multiple_synapse_presynapses': len(multiple_synapse_presynapses),
                                'max_synapses_per_presynapse': presynapse_counts.max() if not presynapse_counts.empty else 0,
                                'presynapse_counts': presynapse_counts.to_dict()
                            }
                    except Exception as e:
                        logger.error(f"Error analyzing presynapse data for seg_type={seg_type}, alpha={alpha}: {e}")
        
        # Create the HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Presynapse Analysis Summary - {self.timestamp}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.css_styles}
        </head>
        <body>
            <h1>Presynapse Analysis Summary Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <h2>Presynapse Analysis Summary</h2>
                <p>This report summarizes presynapse analysis results across all segmentation types and alpha values.</p>
                
                <table>
                    <tr>
                        <th>Segmentation Type</th>
                        <th>Alpha Value</th>
                        <th>Total Presynapses</th>
                        <th>Presynapses with Multiple Synapses</th>
                        <th>Max Synapses per Presynapse</th>
                    </tr>
        """
        
        # Add a row for each combination with presynapse data
        for (seg_type, alpha), data in sorted(presynapse_data.items()):
            html += f"""
                <tr>
                    <td>{seg_type}</td>
                    <td>{alpha}</td>
                    <td>{data['total_presynapses']}</td>
                    <td>{data['multiple_synapse_presynapses']}</td>
                    <td>{data['max_synapses_per_presynapse']}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
            
            <h2>Presynapse Details by Segmentation Type</h2>
        """
        
        # Add detailed sections for each combo with presynapse data
        for (seg_type, alpha), data in sorted(presynapse_data.items()):
            html += f"""
            <div class="card">
                <div class="card-header">
                    <h3>Segmentation Type {seg_type} with Alpha {alpha}</h3>
                </div>
                
                <h4>Presynapses with Multiple Synapses</h4>
                <table>
                    <tr>
                        <th>Presynapse ID</th>
                        <th>Number of Synapses</th>
                    </tr>
            """
            
            # Add rows for presynapses with multiple synapses
            for pre_id, count in sorted(data['presynapse_counts'].items()):
                if count > 1:
                    html += f"""
                    <tr>
                        <td>{pre_id}</td>
                        <td>{count}</td>
                    </tr>
                    """
            
            html += """
                </table>
                
                <!-- Include visualizations if available -->
            """
            
            # Add connected UMAP visualization if available
            viz = self.visualizations.get((seg_type, alpha), {})
            if 'connected_umap' in viz:
                asset_path = self.copy_image_to_assets(viz['connected_umap'], report_dir)
                html += f"""
                <h4>Connected UMAP Visualization</h4>
                <img src="{asset_path}" alt="Connected UMAP" style="max-width: 100%;">
                """
            
            html += """
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Write the report to file
        report_path = report_dir / "presynapse_summary.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Presynapse summary report generated at {report_path}")
        return str(report_path)


if __name__ == "__main__":
    # Create the report generator
    report_generator = SynapseReportGenerator()
    
    # Generate the complete report
    report_path = report_generator.generate_complete_report()
    
    # Generate the presynapse summary
    presynapse_summary_path = report_generator.generate_presynapse_summary()
    
    if report_path:
        print(f"Report generated at: {report_path}")
        print(f"Open the report in your browser to view the results.")
    
    if presynapse_summary_path:
        print(f"Presynapse summary report generated at: {presynapse_summary_path}") 