import os
import sys
import argparse
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from jinja2 import Template
import logging
from typing import List, Tuple, Optional
from pathlib import Path

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_features(features_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load features from CSV file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading features from {features_path}")
    
    df = pd.read_csv(features_path)
    features = df.drop('bbox_name', axis=1).values
    bbox_names = df['bbox_name'].tolist()
    
    logger.info(f"Loaded {len(bbox_names)} samples with {features.shape[1]} features each")
    return features, bbox_names

def get_gif_paths(bbox_names: List[str], gif_dir: Optional[str] = None) -> List[str]:
    """Get paths to GIF files for each bounding box."""
    logger = logging.getLogger(__name__)
    
    if gif_dir is None:
        logger.info("No GIF directory specified, skipping GIF paths")
        return [""] * len(bbox_names)
    
    gif_paths = []
    for bbox_name in bbox_names:
        gif_path = os.path.join(gif_dir, f"{bbox_name}.gif")
        if os.path.exists(gif_path):
            gif_paths.append(gif_path)
        else:
            logger.warning(f"GIF not found for {bbox_name}")
            gif_paths.append("")
    
    logger.info(f"Found {sum(1 for p in gif_paths if p)} GIFs out of {len(bbox_names)} samples")
    return gif_paths

def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualization with GIFs")
    parser.add_argument("--features", type=str, required=True, help="Path to features CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output HTML file path")
    parser.add_argument("--gif-dir", type=str, help="Directory containing GIF files")
    parser.add_argument("--title", type=str, default="UMAP Visualization", help="Title for the visualization")
    parser.add_argument("--n-components", type=int, default=2, help="Number of UMAP components")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--min-dist", type=float, default=0.1, help="Minimum distance for UMAP")
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric for UMAP")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Load features and bounding box names
    features, bbox_names = load_features(args.features)
    
    # Get GIF paths if directory is specified
    gif_paths = get_gif_paths(bbox_names, args.gif_dir)
    
    # Fit UMAP
    logger.info("Fitting UMAP...")
    umap_reducer = umap.UMAP(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=42
    )
    umap_embeddings = umap_reducer.fit_transform(features)
    
    # Create plot data
    plot_data = [{
        'type': 'scatter',
        'mode': 'markers',
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'text': bbox_names,
        'hoverinfo': 'text',
        'marker': {
            'size': 10,
            'color': 'blue',
            'opacity': 0.7
        }
    }]
    
    # Create plot layout
    plot_layout = {
        'title': args.title,
        'showlegend': False,
        'hovermode': 'closest',
        'xaxis': {'title': 'UMAP 1'},
        'yaxis': {'title': 'UMAP 2'}
    }
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), 'template.html')
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    # Render template
    html_content = template.render(
        title=args.title,
        plot_data=plot_data,
        plot_layout=plot_layout,
        bbox_names=bbox_names,
        gif_paths=gif_paths
    )
    
    # Save output
    logger.info(f"Saving visualization to {args.output}")
    with open(args.output, 'w') as f:
        f.write(html_content)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 