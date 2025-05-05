import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Paths to the visualization folders for each pooling method
base_dir = "results/22april"
pooling_methods = ["visualizations_avg", "visualizations_max", "visualizations_concat"]

# Descriptions for each pooling method to add as annotations
pooling_descriptions = {
    "avg": "Average pooling: Smooth, stable features\nthat capture overall patterns",
    "max": "Max pooling: Distinctive features\nthat highlight strongest activations",
    "concat": "Concatenated pooling: Combined avg+max\nfor richer feature representation"
}

def create_comparison_image(image_name, output_filename):
    """Create a comparison image for the specified image type across pooling methods"""
    fig, axes = plt.subplots(1, len(pooling_methods), figsize=(18, 6))
    
    for i, method in enumerate(pooling_methods):
        # Extract the pooling method name from the directory name
        pooling_name = method.replace('visualizations_', '')
        
        image_path = os.path.join(base_dir, method, image_name)
        try:
            img = Image.open(image_path)
            axes[i].imshow(np.array(img))
            axes[i].set_title(f"{pooling_name.upper()} Pooling", fontsize=14)
            
            # Add descriptive annotation if available
            if pooling_name in pooling_descriptions:
                axes[i].annotate(
                    pooling_descriptions[pooling_name],
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7),
                    ha='center', va='top', fontsize=10
                )
                
            axes[i].axis('off')
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}")
            axes[i].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i].axis('off')
    
    # Add a title for the entire figure
    if "bbox" in image_name:
        fig.suptitle("UMAP Visualization Comparison: Colored by Bounding Box", fontsize=16, y=0.98)
    elif "cluster" in image_name:
        fig.suptitle("UMAP Visualization Comparison: Colored by Cluster", fontsize=16, y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(base_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison image to {output_path}")
    plt.close()

# Create comparison images
create_comparison_image("umap_bbox_colored.png", "comparison_umap_bbox.png")
create_comparison_image("umap_cluster_colored.png", "comparison_umap_cluster.png")

print("Comparison images created successfully!") 