import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

# Load the features from CSV file
def generate_umap_plot(csv_filepath, seg_type=9, alpha=1.0):
    print(f"Loading features from {csv_filepath}")
    features_df = pd.read_csv(csv_filepath)
    
    # Process features and compute UMAP
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    features = features_df[feature_cols].values
    print(f"Features shape: {features.shape}")
    
    features_scaled = StandardScaler().fit_transform(features)
    print("Features scaled")
    
    # Compute UMAP
    print("Computing UMAP")
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(features_scaled)
    print(f"UMAP results shape: {umap_results.shape}")
    
    # Add UMAP coordinates to DataFrame
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    
    # Define a color map for the bounding boxes
    bbox_names = features_df['bbox_name'].unique()
    colors = ['red', 'cyan', 'orange', 'purple', 'gray', 'blue', 'black']
    color_map = {bbox: color for bbox, color in zip(bbox_names, colors)}
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a scatter plot
    for bbox in bbox_names:
        bbox_data = features_df[features_df['bbox_name'] == bbox]
        plt.scatter(
            bbox_data['umap_x'], 
            bbox_data['umap_y'],
            color=color_map[bbox],
            label=bbox,
            alpha=0.7,
            s=50
        )
    
    # Add legend and title
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"VGG Segmentation Type {seg_type} (Alpha={alpha})", fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    # Save the figure
    output_filename = f"umap_plot_seg{seg_type}_alpha{str(alpha).replace('.', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP plot to {output_filename}")
    
    # Also show it
    plt.show()
    
    return output_filename

if __name__ == "__main__":
    # Path to the CSV file with features
    csv_filepath = "csv_outputs/features_seg9_alpha1.csv"
    
    # Generate the UMAP plot
    output_file = generate_umap_plot(csv_filepath)
    print(f"UMAP visualization saved to {output_file}") 