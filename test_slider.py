import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

# Add the current directory to Python path
sys.path.append('.')

# Import the interactive plot function
from compare_csvs import create_interactive_plot

def extract_features_and_ids(csv_path):
    """Extract features and sample IDs from a CSV file"""
    print(f"Loading features from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Figure out the feature columns
    feature_cols = []
    
    # Try layer20_feat_ prefix (from stage-specific extraction)
    layer_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
    if layer_cols:
        feature_cols = layer_cols
        print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
    else:
        # Try feat_ prefix (from standard extraction)
        feat_cols = [col for col in df.columns if col.startswith('feat_')]
        if feat_cols:
            feature_cols = feat_cols
            print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
    
    # If still no feature columns, try to infer from numeric columns
    if not feature_cols:
        non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
    
    if not feature_cols:
        print(f"No feature columns found in {csv_path}")
        return None, None
    
    # Extract features
    features = df[feature_cols].values
    
    # Get sample identifiers
    if 'Var1' in df.columns:
        ids = df['Var1'].tolist()
    elif 'id' in df.columns:
        ids = df['id'].tolist()
    else:
        # Create arbitrary ids based on row number
        ids = [f"sample_{i}" for i in range(len(df))]
    
    return features, ids

# Set output file path
output_file = os.path.join('results', 'method_comparison', 'extraction_method_comparison_umap_interactive.html')

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Find CSV files - this is just a simplified version for testing
# In a real scenario, you would use the actual CSV paths from feature_extraction steps
import glob
csv_files = glob.glob(os.path.join('results', '**', '*.csv'), recursive=True)

if len(csv_files) < 2:
    print("Not enough CSV files found. Using random data instead.")
    # Create sample UMAP data
    umap_data = np.random.rand(200, 2)
    # Generate sample pairs (connecting first 100 points to the second 100)
    sample_pairs = [(i, i+100) for i in range(100)]
else:
    # Use the first two CSV files found
    feature_csv_paths = csv_files[:2]
    print(f"Using CSV files: {feature_csv_paths}")
    
    # Load feature data
    feature_sets = []
    sample_ids = []
    
    for csv_path in feature_csv_paths:
        features, ids = extract_features_and_ids(csv_path)
        if features is not None and ids is not None:
            feature_sets.append(features)
            sample_ids.append(ids)
    
    if len(feature_sets) != 2:
        print("Could not extract features from enough CSV files. Using random data instead.")
        # Create sample UMAP data
        umap_data = np.random.rand(200, 2)
        # Generate sample pairs (connecting first 100 points to the second 100)
        sample_pairs = [(i, i+100) for i in range(100)]
    else:
        # Create UMAP projections
        
        # Check if feature dimensions are the same
        if feature_sets[0].shape[1] != feature_sets[1].shape[1]:
            print(f"Feature dimensions don't match: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}")
            print("Using separate UMAP projections")
            
            # Scale each feature set separately
            scaled_sets = []
            for features in feature_sets:
                scaler = StandardScaler()
                scaled_sets.append(scaler.fit_transform(features))
            
            # Create separate UMAP projections
            reducer = umap.UMAP(random_state=42)
            embedding_1 = reducer.fit_transform(scaled_sets[0])
            
            reducer = umap.UMAP(random_state=42)
            embedding_2 = reducer.fit_transform(scaled_sets[1])
        else:
            # If dimensions match, combine features for UMAP
            combined_features = np.vstack([feature_sets[0], feature_sets[1]])
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(combined_features)
            
            # Create UMAP projection
            print("Computing UMAP embedding...")
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(scaled_features)
            
            # Split embedding back into the two sets
            n_samples_1 = feature_sets[0].shape[0]
            embedding_1 = embedding[:n_samples_1]
            embedding_2 = embedding[n_samples_1:]
        
        # Combine embeddings for visualization
        umap_data = np.vstack([embedding_1, embedding_2])
        
        # Find common samples to create pairs
        set1 = set(sample_ids[0])
        set2 = set(sample_ids[1])
        common_ids = set1.intersection(set2)
        
        if not common_ids:
            print("No common samples found between the two feature sets")
            # If no common IDs, create default pairs
            sample_pairs = [(i, i+len(embedding_1)) for i in range(min(100, len(embedding_1)))]
        else:
            print(f"Found {len(common_ids)} common samples between the two feature sets")
            
            # Create dictionaries to map sample IDs to row indices
            id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
            id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
            
            # Create sample pairs for visualization
            sample_pairs = []
            for sample_id in common_ids:
                idx1 = id_to_idx_1.get(sample_id)
                idx2 = id_to_idx_2.get(sample_id)
                
                if idx1 is not None and idx2 is not None:
                    # For the stacked array, we need to adjust idx2
                    adjusted_idx2 = idx2 + len(embedding_1)
                    sample_pairs.append((idx1, adjusted_idx2))

# Create interactive visualization with slider
print(f"Creating interactive visualization at {output_file}")
create_interactive_plot(
    umap_data=umap_data,
    sample_pairs=sample_pairs,
    output_file=output_file,
    max_pairs=min(100, len(sample_pairs))  # Limit to 100 pairs for performance
)
print("Done!") 