import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import base64
from PIL import Image
import io
import torch
from PIL import ImageDraw

# Set environment variable to use CPU only
os.environ['UMAP_USE_GPU'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import umap
from typing import List, Dict, Tuple, Optional
import logging

class GifUmapContrastive:
    """Class for creating interactive UMAP visualizations with GIFs for contrastive features."""
    
    def __init__(self, 
                 features_df: pd.DataFrame,
                 output_dir: str,
                 method_name: str = "Contrastive",
                 n_components: int = 2,
                 n_neighbors: int = 15,
                 min_dist: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the GifUmap visualization.
        
        Args:
            features_df: DataFrame containing features and metadata
            output_dir: Directory to save visualization files
            method_name: Name of the method for visualization title
            n_components: Number of components for UMAP (2 or 3)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random state for reproducibility
        """
        self.features_df = features_df
        self.output_dir = Path(output_dir)
        self.method_name = method_name
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Extract features and metadata
        self.features, self.feature_cols = self._extract_features()
        self.bbox_names = self.features_df['bbox_name'].tolist()
        self.sample_names = self.features_df['Var1'].tolist()
        
        # Perform UMAP
        self.embeddings = self._perform_umap()
        
        # Create visualization data
        self.visualization_data = self._create_visualization_data()
        
        # Create sample GIFs for visualization
        self.samples_with_gifs = self._create_sample_gifs()
    
    def _extract_features(self) -> Tuple[np.ndarray, List[str]]:
        """Extract feature columns from DataFrame."""
        feature_cols = [col for col in self.features_df.columns if col.startswith('feature_')]
        features = self.features_df[feature_cols].values
        return features, feature_cols
    
    def _perform_umap(self) -> np.ndarray:
        """Perform UMAP dimensionality reduction."""
        self.logger.info(f"Performing UMAP with {self.n_components} components...")
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            low_memory=True  # Add low_memory option for better performance
        )
        embeddings = reducer.fit_transform(self.features)
        return embeddings
    
    def _create_visualization_data(self) -> Dict:
        """Create data for visualization."""
        # Normalize embeddings to [0, 1] range for plotting
        embeddings_norm = (self.embeddings - self.embeddings.min(axis=0)) / (self.embeddings.max(axis=0) - self.embeddings.min(axis=0))
        
        # Create points data
        points = []
        for i, (x, y) in enumerate(embeddings_norm):
            point = {
                'id': i,
                'x': float(x),
                'y': float(y),
                'bbox_name': self.bbox_names[i],
                'sample_name': self.sample_names[i],
                'color': self._get_color_for_bbox(self.bbox_names[i])
            }
            points.append(point)
        
        # Create visualization data
        visualization_data = {
            'method_name': self.method_name,
            'points': points,
            'bbox_colors': {bbox: self._get_color_for_bbox(bbox) 
                          for bbox in sorted(set(self.bbox_names))},
            'x_min': float(self.embeddings[:, 0].min()),
            'x_max': float(self.embeddings[:, 0].max()),
            'y_min': float(self.embeddings[:, 1].min()),
            'y_max': float(self.embeddings[:, 1].max()),
            'segmentation_type': 'contrastive',
            'originalPositions': {}
        }
        
        return visualization_data
    
    def _get_color_for_bbox(self, bbox_name: str) -> str:
        """Get color for a bounding box."""
        # Extract bbox number
        bbox_num = int(bbox_name.replace('bbox', ''))
        
        # Use a consistent color map
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        color = colors[bbox_num % 20]
        
        # Convert to hex
        return '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255)
        )
    
    def _create_sample_gifs(self) -> List[Dict]:
        """Create sample GIFs for visualization."""
        # Create a few sample GIFs for demonstration
        samples = []
        
        # Select a few points from each bbox
        unique_bboxes = sorted(set(self.bbox_names))
        for bbox in unique_bboxes:
            # Get indices for this bbox
            bbox_indices = [i for i, name in enumerate(self.bbox_names) if name == bbox]
            
            # Select up to 3 points from this bbox
            selected_indices = np.random.choice(bbox_indices, min(3, len(bbox_indices)), replace=False)
            
            for idx in selected_indices:
                # Create a sample GIF
                sample = {
                    'id': f"sample_{idx}",
                    'bbox_name': self.bbox_names[idx],
                    'sample_name': self.sample_names[idx],
                    'x': float(self.embeddings[idx, 0]),
                    'y': float(self.embeddings[idx, 1]),
                    'color': self._get_color_for_bbox(self.bbox_names[idx]),
                    'frames': self._create_sample_frames(idx)
                }
                samples.append(sample)
        
        return samples
    
    def _create_sample_frames(self, idx: int) -> List[str]:
        """Create sample frames for a GIF."""
        # Create a simple colored square as a placeholder
        frames = []
        
        # Create 5 frames with different colors
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255)   # Magenta
        ]
        
        for color in colors:
            # Create a 50x50 image with the color
            img = Image.new('RGB', (50, 50), color)
            
            # Add text to the image
            draw = ImageDraw.Draw(img)
            
            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 10)
            except IOError:
                font = ImageFont.load_default()
            
            # Add bbox name and index
            text = f"{self.bbox_names[idx]}\n{idx}"
            draw.text((5, 5), text, fill=(255, 255, 255), font=font)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            frames.append(img_str)
        
        return frames
    
    def save_visualization(self) -> str:
        """Save visualization files."""
        # Save visualization data
        data_path = self.output_dir / 'visualization_data.json'
        with open(data_path, 'w') as f:
            json.dump(self.visualization_data, f)
        
        # Save samples with GIFs
        samples_path = self.output_dir / 'samples_with_gifs.json'
        with open(samples_path, 'w') as f:
            json.dump(self.samples_with_gifs, f)
        
        # Create frames data
        frames_data = {}
        for sample in self.samples_with_gifs:
            frames_data[sample['id']] = sample['frames']
        
        # Save frames data
        frames_path = self.output_dir / 'frames_data.json'
        with open(frames_path, 'w') as f:
            json.dump(frames_data, f)
        
        # Copy and customize template
        template_path = Path(__file__).parent / 'template.html'
        output_path = self.output_dir / 'index.html'
        
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace placeholders
        html_content = template.replace("{method_name}", self.method_name)
        html_content = html_content.replace("{len(samples_with_gifs)}", str(len(self.samples_with_gifs)))
        html_content = html_content.replace("{x_min}", str(self.visualization_data['x_min']))
        html_content = html_content.replace("{x_max}", str(self.visualization_data['x_max']))
        html_content = html_content.replace("{y_min}", str(self.visualization_data['y_min']))
        html_content = html_content.replace("{y_max}", str(self.visualization_data['y_max']))
        html_content = html_content.replace("{segmentation_type}", f"'{self.visualization_data['segmentation_type']}'")
        html_content = html_content.replace("{originalPositions}", json.dumps(self.visualization_data['originalPositions']))
        
        # Replace frames content
        frames_content = json.dumps(frames_data)
        html_content = html_content.replace("{frames_content}", frames_content)
        
        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)

    def extract_features(self, model, data_loader):
        """Extract features from the model."""
        features = []
        model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                    
                if hasattr(model, 'forward') and 'return_features' in model.forward.__code__.co_varnames:
                    outputs, batch_features = model(inputs, return_features=True)
                    features.append(batch_features.cpu().numpy())
                else:
                    outputs = model(inputs)
                    features.append(outputs.cpu().numpy())
                    
        return np.concatenate(features, axis=0)
        
    def perform_umap(self, features, n_components=2):
        """Perform UMAP dimensionality reduction."""
        reducer = umap.UMAP(n_components=n_components)
        embedding = reducer.fit_transform(features)
        return embedding
        
    def _create_sample_gifs(self, samples, n_frames=10):
        """Create sample GIFs for visualization."""
        gifs = {}
        for i, sample in enumerate(samples):
            frames = []
            for j in range(n_frames):
                # Create a frame with a colored square and text
                img = Image.new('RGB', (64, 64), color='white')
                draw = ImageDraw.Draw(img)
                
                # Draw a colored square
                color = sample.get('color', '#FF0000')
                draw.rectangle([10, 10, 54, 54], fill=color)
                
                # Add sample index
                draw.text((20, 25), str(i), fill='white')
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                frames.append(base64.b64encode(buffered.getvalue()).decode())
                
            gifs[f'sample_{i}'] = frames
            
        return gifs
        
    def _create_visualization_data(self, embedding, samples, bbox_names):
        """Create visualization data including points, colors, and GIFs."""
        # Create points data
        points = []
        bbox_colors = {}
        for i, (x, y) in enumerate(embedding):
            bbox_name = bbox_names[i]
            if bbox_name not in bbox_colors:
                bbox_colors[bbox_name] = f'#{hash(bbox_name) & 0xFFFFFF:06x}'
                
            points.append({
                'x': float(x),
                'y': float(y),
                'color': bbox_colors[bbox_name],
                'bbox': bbox_name
            })
            
        # Create sample GIFs
        samples_with_gifs = []
        for i, sample in enumerate(samples):
            samples_with_gifs.append({
                'id': f'sample_{i}',
                'x': float(embedding[i, 0]),
                'y': float(embedding[i, 1]),
                'bbox': bbox_names[i]
            })
            
        # Calculate UMAP ranges
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
        
        # Create visualization data
        visualization_data = {
            'points': points,
            'bbox_colors': bbox_colors,
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max),
            'samples_with_gifs': samples_with_gifs
        }
        
        # Add GIF frames
        gifs = self._create_sample_gifs(samples_with_gifs)
        visualization_data.update(gifs)
        
        return visualization_data
        
    def save_visualization(self, visualization_data, template_path):
        """Save the visualization using the template."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save visualization data
        data_path = os.path.join(self.output_dir, 'visualization_data.json')
        with open(data_path, 'w') as f:
            json.dump(visualization_data, f)
            
        # Read template
        with open(template_path, 'r') as f:
            template = f.read()
            
        # Replace placeholders
        html_content = template.format(
            method_name='Contrastive Learning',
            frames_content=json.dumps(visualization_data),
            len=len,
            samples_with_gifs=visualization_data['samples_with_gifs'],
            x_min=visualization_data['x_min'],
            x_max=visualization_data['x_max'],
            y_min=visualization_data['y_min'],
            y_max=visualization_data['y_max'],
            segmentation_type='synapse',
            originalPositions=json.dumps([[p['x'], p['y']] for p in visualization_data['points']])
        )
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, 'umap_visualization.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Visualization saved to {output_path}")
        return output_path 