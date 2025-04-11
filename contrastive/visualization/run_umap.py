import os
import sys
import torch
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ContrastiveConfig
from models.contrastive import VGG3DContrastive
from data.dataset import SynapseDataset
from torch.utils.data import DataLoader
from gif_umap.GifUmapContrastive import GifUmapContrastive

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model(checkpoint_path, config):
    """Load the trained model from checkpoint."""
    model = VGG3DContrastive(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting UMAP visualization")
    
    # Parse arguments
    config = ContrastiveConfig()
    args = config.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / 'visualization' / 'umap'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config)
    model.eval()
    
    # Create dataset and dataloader
    dataset = SynapseDataset(
        data_dir=args.data_dir,
        transform=None  # No transforms needed for visualization
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create UMAP visualizer
    visualizer = GifUmapContrastive(output_dir)
    
    # Extract features
    logger.info("Extracting features")
    features = visualizer.extract_features(model, dataloader)
    
    # Perform UMAP
    logger.info("Performing UMAP dimensionality reduction")
    embedding = visualizer.perform_umap(features)
    
    # Create visualization data
    logger.info("Creating visualization data")
    samples = dataset.samples  # Get the samples from the dataset
    bbox_names = [sample['bbox_name'] for sample in samples]
    visualization_data = visualizer._create_visualization_data(embedding, samples, bbox_names)
    
    # Save visualization
    template_path = Path(__file__).parent / 'gif_umap' / 'template.html'
    output_path = visualizer.save_visualization(visualization_data, template_path)
    
    logger.info(f"Visualization saved to {output_path}")
    logger.info("UMAP visualization completed successfully")

if __name__ == '__main__':
    main() 