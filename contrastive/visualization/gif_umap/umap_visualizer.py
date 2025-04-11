import os
import numpy as np
import umap
import plotly.graph_objects as go
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional
import torch
from tqdm import tqdm

class GifUmapContrastive:
    def __init__(
        self,
        output_dir: str,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 42
    ):
        """
        Initialize the UMAP visualizer for contrastive features.
        
        Args:
            output_dir: Directory to save visualizations
            n_components: Number of UMAP components (2 or 3)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points in UMAP
            metric: Distance metric for UMAP
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        self.logger = logging.getLogger(__name__)
        
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit UMAP to features and transform them.
        
        Args:
            features: Array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_components)
        """
        self.logger.info(f"Fitting UMAP to {len(features)} samples...")
        return self.reducer.fit_transform(features)
    
    def create_visualization(
        self,
        points: np.ndarray,
        bbox_names: List[str],
        colors: Optional[np.ndarray] = None,
        gifs: Optional[List[str]] = None,
        title: str = "Synapse UMAP Visualization"
    ) -> Dict:
        """
        Create visualization data for the UMAP plot.
        
        Args:
            points: Array of shape (n_samples, n_components)
            bbox_names: List of bounding box names
            colors: Optional array of colors for points
            gifs: Optional list of GIF paths
            title: Plot title
            
        Returns:
            Dictionary containing visualization data
        """
        if colors is None:
            colors = np.zeros(len(points))  # Default color
            
        # Calculate ranges for axes
        x_range = [points[:, 0].min(), points[:, 0].max()]
        y_range = [points[:, 1].min(), points[:, 1].max()]
        
        if self.n_components == 3:
            z_range = [points[:, 2].min(), points[:, 2].max()]
            ranges = {'x': x_range, 'y': y_range, 'z': z_range}
        else:
            ranges = {'x': x_range, 'y': y_range}
            
        return {
            'points': points.tolist(),
            'bbox_names': bbox_names,
            'colors': colors.tolist(),
            'gifs': gifs,
            'umap_ranges': ranges,
            'title': title
        }
    
    def save_visualization(
        self,
        vis_data: Dict,
        template_path: str,
        output_name: str = "umap_visualization.html"
    ):
        """
        Save the visualization to an HTML file.
        
        Args:
            vis_data: Visualization data dictionary
            template_path: Path to HTML template
            output_name: Name of output file
        """
        output_path = self.output_dir / output_name
        
        # Read template
        with open(template_path, 'r') as f:
            template = f.read()
            
        # Insert data
        html_content = template.replace('{{data}}', json.dumps(vis_data))
        
        # Save file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Saved visualization to {output_path}")
        
    def create_plotly_figure(
        self,
        points: np.ndarray,
        bbox_names: List[str],
        colors: Optional[np.ndarray] = None,
        title: str = "Synapse UMAP Visualization"
    ) -> go.Figure:
        """
        Create a Plotly figure for the UMAP visualization.
        
        Args:
            points: Array of shape (n_samples, n_components)
            bbox_names: List of bounding box names
            colors: Optional array of colors for points
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if colors is None:
            colors = np.zeros(len(points))
            
        if self.n_components == 3:
            trace = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.7
                ),
                text=bbox_names,
                hoverinfo='text'
            )
        else:
            trace = go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.7
                ),
                text=bbox_names,
                hoverinfo='text'
            )
            
        layout = go.Layout(
            title=title,
            showlegend=False
        )
        
        return go.Figure(data=[trace], layout=layout)
    
    def save_plotly_figure(
        self,
        fig: go.Figure,
        output_name: str = "umap_plotly.html"
    ):
        """
        Save a Plotly figure to an HTML file.
        
        Args:
            fig: Plotly figure object
            output_name: Name of output file
        """
        output_path = self.output_dir / output_name
        fig.write_html(str(output_path))
        self.logger.info(f"Saved Plotly figure to {output_path}") 