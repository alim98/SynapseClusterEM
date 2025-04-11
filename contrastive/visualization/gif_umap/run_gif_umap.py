import pandas as pd
import numpy as np
import umap
import os
import logging
from pathlib import Path
from GifUmapContrastive import GifUmapContrastive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_features(csv_path):
    """Load features from CSV file and separate features from metadata."""
    logger.info(f"Loading features from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Get feature columns (those starting with 'feature_')
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Create a new DataFrame with features and required metadata
    features_df = pd.DataFrame({
        'bbox_name': df['bbox_name'],
        'Var1': df['Var1'],
        **{col: df[col] for col in feature_cols}
    })
    
    return features_df

def main():
    # Load features and metadata
    features_df = load_features('contrastive/results/contrastive_features.csv')
    
    # Create output directory
    output_dir = Path('contrastive/results/umap_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    logger.info("Creating GIF UMAP visualization...")
    gif_umap = GifUmapContrastive(
        features_df=features_df,
        output_dir=output_dir,
        method_name="Contrastive Learning"
    )
    
    # Save visualization
    template_path = os.path.join(os.path.dirname(__file__), 'template.html')
    output_path = gif_umap.save_visualization()
    
    logger.info(f"Visualization saved to {output_path}")
    logger.info("Open the HTML file in a web browser to view the visualization")

if __name__ == '__main__':
    main() 