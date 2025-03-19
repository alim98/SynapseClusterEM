"""
Synapse Analysis Pipeline - Main Runner

This script orchestrates the synapse analysis workflow by:
1. Running feature extraction with stage_specified layer 20 and intelligent cropping
2. Running feature extraction with standard feature extraction and intelligent cropping
3. Running clustering on the resulting CSV files
4. Running projection on the clustered CSV files
5. Creating a comparative UMAP visualization of the two preprocessing methods
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the required modules
from feature_extraction.FeatureExtraction import FeatureExtraction
from cluster.clustering import perform_clustering
import projection

def create_comparative_umap(feature_csv_paths, output_dir):
    """
    Create a UMAP visualization comparing features extracted with different preprocessing methods.
    Points from the same sample are connected with lines to visualize the effect of preprocessing.
    
    Args:
        feature_csv_paths: List of paths to feature CSV files from different preprocessing methods
        output_dir: Directory to save the visualization
    """
    print("\n" + "="*80)
    print("Creating comparative UMAP visualization...")
    
    if len(feature_csv_paths) != 2:
        print(f"Expected 2 feature CSV files for comparison, got {len(feature_csv_paths)}")
        return
    
    # Load feature data from both preprocessing methods
    dataframes = []
    feature_sets = []
    sample_ids = []
    
    for i, csv_path in enumerate(feature_csv_paths):
        print(f"Loading features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Figure out the feature columns
        feature_cols = []
        # First try layer20_feat_ prefix (from stage-specific extraction)
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
            else:
                # Try other common prefixes
                for prefix in ['feature_', 'f_', 'layer']:
                    cols = [col for col in df.columns if col.startswith(prefix)]
                    if cols:
                        feature_cols = cols
                        print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                        break
        
        # If still no feature columns, try to infer from numeric columns
        if not feature_cols:
            non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
        
        if not feature_cols:
            print(f"No feature columns found in {csv_path}")
            return
        
        print(f"Found {len(feature_cols)} feature columns")
        
        # Extract features
        features = df[feature_cols].values
        
        # Get sample identifiers
        # First check if Var1 column exists (synapse identifier)
        if 'Var1' in df.columns:
            ids = df['Var1'].tolist()
        elif 'id' in df.columns:
            ids = df['id'].tolist()
        else:
            # Create arbitrary ids based on row number
            ids = [f"sample_{i}" for i in range(len(df))]
        
        # Store data
        dataframes.append(df)
        feature_sets.append(features)
        sample_ids.append(ids)
        
        # Add preprocessing method tag
        if i == 0:
            # Assume first is stage-specific
            df['preprocessing'] = 'Stage-Specific Layer 20'
        else:
            # Assume second is standard
            df['preprocessing'] = 'Standard Extraction'
    
    # Check if both datasets have the same sample ids
    set1 = set(sample_ids[0])
    set2 = set(sample_ids[1])
    common_ids = set1.intersection(set2)
    
    if not common_ids:
        print("No common samples found between the two feature sets")
        # If no common IDs, we'll create a UMAP plot without connecting lines
    else:
        print(f"Found {len(common_ids)} common samples between the two feature sets")
    
    # Check if feature dimensions are the same
    if feature_sets[0].shape[1] != feature_sets[1].shape[1]:
        print(f"Feature dimensions don't match: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}")
        print("Using separate UMAP projections and aligning them to compare the feature spaces")
        
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
        # Combine features for UMAP if they have the same dimensions
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
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(embedding_1[:, 0], embedding_1[:, 1], c='blue', label='Stage-Specific Layer 20', alpha=0.7)
    plt.scatter(embedding_2[:, 0], embedding_2[:, 1], c='red', label='Standard Extraction', alpha=0.7)
    
    # If we have common samples, draw connecting lines
    if common_ids:
        # Create dictionaries to map sample IDs to row indices
        id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
        id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
        
        # Draw lines connecting the same samples
        for sample_id in common_ids:
            idx1 = id_to_idx_1.get(sample_id)
            idx2 = id_to_idx_2.get(sample_id)
            
            if idx1 is not None and idx2 is not None:
                x_values = [embedding_1[idx1, 0], embedding_2[idx2, 0]]
                y_values = [embedding_1[idx1, 1], embedding_2[idx2, 1]]
                plt.plot(x_values, y_values, 'k-', alpha=0.3)
    
    plt.title('UMAP Comparison of Stage-Specific vs Standard Feature Extraction\n(Both with Intelligent Cropping)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'extraction_method_comparison_umap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative UMAP visualization saved to {output_path}")
    
    # Create an interactive HTML version with plotly if available
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Try to import our enhanced interactive plotting function
        try:
            # Add project root to path to ensure module imports work
            sys.path.append(project_root)
            from compare_csvs import create_interactive_plot
            
            # Create a combined UMAP dataset
            combined_umap = np.vstack([embedding_1, embedding_2])
            
            # Create sample pairs for visualization
            viz_sample_pairs = []
            if common_ids:
                for sample_id in common_ids:
                    idx1 = id_to_idx_1.get(sample_id)
                    idx2 = id_to_idx_2.get(sample_id)
                    
                    if idx1 is not None and idx2 is not None:
                        # For the stacked array, we need to adjust idx2
                        adjusted_idx2 = idx2 + len(embedding_1)
                        viz_sample_pairs.append((idx1, adjusted_idx2))
            
            # Generate the enhanced interactive visualization with slider
            html_path = os.path.join(output_dir, 'extraction_method_comparison_umap_interactive.html')
            create_interactive_plot(
                umap_data=combined_umap,
                sample_pairs=viz_sample_pairs,
                output_file=html_path,
                max_pairs=min(100, len(viz_sample_pairs))  # Limit to 100 pairs for performance
            )
            print(f"Enhanced interactive visualization with slider control saved to {html_path}")
            
        except (ImportError, Exception) as e:
            print(f"Could not use enhanced plotting function: {e}")
            print("Falling back to standard interactive visualization")
            
            # If the enhanced plotting fails, use the original approach
            # Create a combined dataframe for plotly
            df1 = dataframes[0].copy()
            df2 = dataframes[1].copy()
            
            # Add UMAP coordinates
            df1['umap_x'] = embedding_1[:, 0]
            df1['umap_y'] = embedding_1[:, 1]
            df2['umap_x'] = embedding_2[:, 0]
            df2['umap_y'] = embedding_2[:, 1]
            
            combined_df = pd.concat([df1, df2])
            
            # Create the scatter plot
            fig = px.scatter(
                combined_df, 
                x='umap_x', 
                y='umap_y', 
                color='preprocessing',
                hover_data=['Var1'] if 'Var1' in combined_df.columns else None,
                title='Interactive UMAP Comparison of Extraction Methods (Both with Intelligent Cropping)'
            )
            
            # Add connecting lines if we have common samples
            if common_ids:
                for sample_id in common_ids:
                    idx1 = id_to_idx_1.get(sample_id)
                    idx2 = id_to_idx_2.get(sample_id)
                    
                    if idx1 is not None and idx2 is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[embedding_1[idx1, 0], embedding_2[idx2, 0]],
                                y=[embedding_1[idx1, 1], embedding_2[idx2, 1]],
                                mode='lines',
                                line=dict(color='gray', width=0.5),
                                showlegend=False,
                                hoverinfo='none'
                            )
                        )
            
            # Update layout
            fig.update_layout(
                width=900,
                height=700,
                template='plotly_white'
            )
            
            # Save as HTML
            html_path = os.path.join(output_dir, 'extraction_method_comparison_umap_interactive.html')
            fig.write_html(html_path)
            print(f"Interactive UMAP visualization saved to {html_path}")
            
    except ImportError:
        print("Plotly not available, skipping interactive visualization")

def main():
    """Main function to run the complete synapse analysis pipeline"""
    print("Starting Synapse Analysis Pipeline...")
    
    # Create output directory for this run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(project_root, "results", f"pipeline_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    
    # List to store all feature CSV paths
    feature_csv_paths = []
    clustered_csv_paths = []
    
    # 1. Run feature extraction with stage_specific layer 20 and intelligent cropping
    print("\n" + "="*80)
    print("STEP 1: Running Feature Extraction with stage_specific layer 20 and intelligent cropping...")
    
    # Create and configure feature extractor
    extractor1 = FeatureExtraction()
    # Set config parameters for intelligent cropping
    extractor1.config.preprocessing = 'intelligent_cropping'
    extractor1.config.preprocessing_weights = 7
    
    extractor1.create_results_directory()
    extractor1.load_data()
    extractor1.load_model()
    
    # Extract features with stage-specific extraction and intelligent cropping
    features_df1 = extractor1.extract_features(
        extraction_method='stage_specific',
        layer_num=20
    )
    
    # Get the path to the generated CSV
    stage_specific_dir = extractor1.results_parent_dir
    stage_specific_csvs = glob.glob(os.path.join(stage_specific_dir, "**", "*.csv"), recursive=True)
    
    if stage_specific_csvs:
        feature_csv_paths.extend(stage_specific_csvs)
        print(f"Found {len(stage_specific_csvs)} feature CSV files from stage-specific extraction")
    else:
        print("Warning: No feature CSV files found from stage-specific step")
    
    # 2. Run feature extraction with standard method and intelligent cropping
    print("\n" + "="*80)
    print("STEP 2: Running Feature Extraction with standard method and intelligent cropping...")
    
    # Create and configure feature extractor
    extractor2 = FeatureExtraction()
    # Set config parameters for intelligent cropping
    extractor2.config.preprocessing = 'intelligent_cropping'
    extractor2.config.preprocessing_weights = 7
    
    extractor2.create_results_directory()
    extractor2.load_data()
    extractor2.load_model()
    
    # Extract features with standard extraction method and intelligent cropping
    features_df2 = extractor2.extract_features(
        extraction_method='standard'  # Use standard extraction instead of stage_specific
    )
    
    # Get the path to the generated CSV
    standard_dir = extractor2.results_parent_dir
    standard_csvs = glob.glob(os.path.join(standard_dir, "**", "*.csv"), recursive=True)
    
    if standard_csvs:
        feature_csv_paths.extend(standard_csvs)
        print(f"Found {len(standard_csvs)} feature CSV files from standard extraction")
    else:
        print("Warning: No feature CSV files found from standard extraction step")
    
    # 3. Run clustering on all feature CSVs
    print("\n" + "="*80)
    print(f"STEP 3: Running Clustering on {len(feature_csv_paths)} feature CSV files...")
    
    for i, csv_path in enumerate(feature_csv_paths):
        print(f"\nProcessing CSV {i+1}/{len(feature_csv_paths)}: {csv_path}")
        
        # Create output directory for clustering results
        csv_basename = os.path.basename(csv_path).replace(".csv", "")
        clustering_output_dir = os.path.join(results_dir, f"clustering_{csv_basename}")
        os.makedirs(clustering_output_dir, exist_ok=True)
        
        # Check the column names in the CSV to determine the feature prefix
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV has {len(df.columns)} columns")
            
            # Look for layer20_feat_ prefix first (for stage-specific extraction)
            feature_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
            if feature_cols:
                print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
                prefix = 'layer20_feat_'
            else:
                # Look for feat_ prefix (for standard extraction)
                feat_cols = [col for col in df.columns if col.startswith('feat_')]
                if feat_cols:
                    feature_cols = feat_cols
                    print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
                    prefix = 'feat_'
                else:
                    # Try other common prefixes if the expected ones aren't found
                    for prefix in ['feature_', 'f_', 'layer']:
                        feature_cols = [col for col in df.columns if col.startswith(prefix)]
                        if feature_cols:
                            print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                            break
            
            # If no columns found with common prefixes, try to detect numeric columns
            if not feature_cols:
                # Exclude common non-feature columns
                non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var']
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
                
                if feature_cols:
                    # Use the common prefix of these columns or just take all if no common prefix
                    print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
                    prefix = os.path.commonprefix(feature_cols) if len(feature_cols) > 1 else ""
                    if not prefix:
                        print("No common prefix found, using all numeric columns as features")
            
            # Perform clustering with the detected prefix
            if feature_cols:
                if prefix:
                    print(f"Using prefix '{prefix}' for clustering")
                    clustered_csv_path = perform_clustering(
                        input_csv_path=csv_path,
                        output_dir=clustering_output_dir,
                        n_clusters=10,
                        algorithm='KMeans',
                        feature_prefix=prefix
                    )
                else:
                    # If no prefix, pass the exact feature column list
                    print("Using explicit feature columns for clustering")
                    # Read the CSV again and manually extract features
                    df = pd.read_csv(csv_path)
                    features = df[feature_cols].values
                    
                    # Apply clustering directly
                    # Modify the clustering.py code to accept explicit feature columns
                    # Since we can't modify that file here, let's create a temporary CSV with renamed columns
                    temp_csv_path = os.path.join(clustering_output_dir, "temp_features.csv")
                    
                    # Rename columns to use feat_ prefix
                    renamed_cols = {col: f'feat_{i}' for i, col in enumerate(feature_cols)}
                    df_temp = df.copy()
                    df_temp.rename(columns=renamed_cols, inplace=True)
                    df_temp.to_csv(temp_csv_path, index=False)
                    
                    # Now cluster using the temporary CSV with proper prefixes
                    clustered_csv_path = perform_clustering(
                        input_csv_path=temp_csv_path,
                        output_dir=clustering_output_dir,
                        n_clusters=10,
                        algorithm='KMeans',
                        feature_prefix='feat_'
                    )
                
                clustered_csv_paths.append(clustered_csv_path)
                print(f"Clustering completed. Output saved to: {clustered_csv_path}")
            else:
                print(f"Error: No suitable feature columns found in {csv_path}")
                continue
        except Exception as e:
            print(f"Error during clustering of {csv_path}: {e}")
    
    # 4. Run projection on all clustered CSVs
    print("\n" + "="*80)
    print(f"STEP 4: Running Projection on {len(clustered_csv_paths)} clustered CSV files...")
    
    for i, csv_path in enumerate(clustered_csv_paths):
        print(f"\nProcessing CSV {i+1}/{len(clustered_csv_paths)}: {csv_path}")
        
        # Create output directory for projection results
        csv_basename = os.path.basename(csv_path).replace(".csv", "")
        projection_output_dir = os.path.join(results_dir, f"projection_{csv_basename}")
        os.makedirs(projection_output_dir, exist_ok=True)
        
        # Run projection
        try:
            # Identify feature columns with the same approach as in clustering
            df = pd.read_csv(csv_path)
            print(f"CSV has {len(df.columns)} columns")
            
            # Look for layer20_feat_ prefix first (for stage-specific extraction)
            feature_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
            if feature_cols:
                print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
            else:
                # Look for feat_ prefix (for standard extraction)
                feat_cols = [col for col in df.columns if col.startswith('feat_')]
                if feat_cols:
                    feature_cols = feat_cols
                    print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
                else:
                    # Look for other potential feature columns
                    for prefix in ['feature_', 'f_', 'layer']:
                        cols = [col for col in df.columns if col.startswith(prefix)]
                        if cols:
                            feature_cols = cols
                            print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                            break
            
            # If no columns found with common prefixes, try to detect numeric columns
            if not feature_cols:
                # Exclude common non-feature columns
                non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var']
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
                
                if feature_cols:
                    print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
            
            # Check if we have cluster assignments
            has_cluster = 'cluster' in df.columns
            has_vesicle_size = 'vesicle_cloud_size' in df.columns
            
            if feature_cols:
                # Create projections
                print(f"Creating projections with {len(feature_cols)} feature columns")
                projection.create_projections(
                    df=df,
                    feature_cols=feature_cols,
                    output_dir=projection_output_dir,
                    has_cluster=has_cluster,
                    has_vesicle_size=has_vesicle_size
                )
                print(f"Projection completed. Output saved to: {projection_output_dir}")
            else:
                print(f"Error: No suitable feature columns found in {csv_path} for projection")
                continue
        except Exception as e:
            print(f"Error during projection of {csv_path}: {e}")
    
    # 5. Create comparative UMAP visualization of the two extraction methods
    comparative_umap_dir = os.path.join(results_dir, "comparative_umap")
    create_comparative_umap(feature_csv_paths, comparative_umap_dir)
    
    print("\n" + "="*80)
    print("Synapse Analysis Pipeline completed!")
    print(f"All results have been saved in: {results_dir}")
    
    return results_dir

if __name__ == "__main__":
    main()

