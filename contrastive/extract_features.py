import torch
import os
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
try:
    from synapse.models import Vgg3D
except ImportError:
    print("Could not import Vgg3D from synapse.models")
    Vgg3D = None

try:
    from contrastive.models.contrastive_model import initialize_contrastive_model, VGG3DContrastive
    from contrastive.utils.config import config
except ImportError as e:
    print(f"Error importing contrastive modules: {e}")

def setup_logging():
    """Set up logging for the feature extraction."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint into model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False

def extract_features_direct(checkpoint_path, output_path, device="cuda"):
    """Extract features directly from a checkpoint file."""
    logger = setup_logging()
    logger.info(f"Extracting features from checkpoint: {checkpoint_path}")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = VGG3DContrastive(
        checkpoint_path=None,  # We'll load the checkpoint manually
        proj_dim=128,
        input_size=(80, 80, 80),
        fmaps=24,
        output_classes=7,
        input_fmaps=1,
        use_pretrained=False  # Don't load pretrained weights automatically
    )
    model = model.to(device)
    
    # Load checkpoint
    if not load_checkpoint(model, checkpoint_path, device):
        logger.error("Failed to extract features due to checkpoint loading error")
        return
    
    model.eval()
    
    # Create a dummy input batch
    dummy_input = torch.randn(5, 1, 80, 80, 80).to(device)
    
    # Extract features
    logger.info("Extracting features from model...")
    with torch.no_grad():
        try:
            # Get features and projections
            features, projections = model(dummy_input, return_features=True)
            features_np = features.cpu().numpy()
            projections_np = projections.cpu().numpy()
            
            # Create feature dictionaries for DataFrame without fragmentation
            data = {
                'sample_id': [f"sample_{i}" for i in range(features_np.shape[0])]
            }
            
            # Add all features at once
            for i in range(features_np.shape[1]):
                data[f'feature_{i}'] = features_np[:, i]
            
            # Add all projections at once
            for i in range(projections_np.shape[1]):
                data[f'projection_{i}'] = projections_np[:, i]
            
            # Create DataFrame in one go
            features_df = pd.DataFrame(data)
            
            # Save features
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            features_df.to_csv(output_path, index=False)
            logger.info(f"Features extracted and saved to: {output_path}")
            
            # Also save numpy arrays for convenience
            np.save(os.path.join(os.path.dirname(output_path), "features.npy"), features_np)
            np.save(os.path.join(os.path.dirname(output_path), "projections.npy"), projections_np)
            logger.info(f"Also saved numpy arrays in {os.path.dirname(output_path)}")
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Extract features from a contrastive model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, default="results/extracted_features.csv", help="Path to save the extracted features")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for feature extraction")
    
    args = parser.parse_args()
    
    # Extract features
    extract_features_direct(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device
    )

if __name__ == "__main__":
    main() 