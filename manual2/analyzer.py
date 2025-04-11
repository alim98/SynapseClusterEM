import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
import sys

# Load required packages with error handling
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        print("Warning: umap-learn is not installed. UMAP visualization will be disabled.")
        print("To install umap, run: pip install umap-learn")
        UMAP_AVAILABLE = False
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn is not installed. Some features will be disabled.")
    print("To install required packages, run: pip install scikit-learn umap-learn")
    SKLEARN_AVAILABLE = False
    UMAP_AVAILABLE = False

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the CSV file
csv_path = os.path.join(script_dir, 'cleaned_df.csv')

def load_data():
    """Load and preprocess the synapse data"""
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Error: File not found at {csv_path}")
            print("Current directory:", os.getcwd())
            print("Files in directory:", os.listdir(script_dir))
            return None
            
        # Load the data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} synapses from {len(df['bbox'].unique())} bboxes")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None

def visualize_feature_distributions(df):
    """Visualize the distribution of categorical features"""
    if df is None:
        print("Cannot visualize feature distributions: No data provided")
        return
        
    try:
        # Select categorical features for visualization
        categorical_features = [
            'vesicle size', 'shape (roundness)', 'Shading inside large vesicles',
            'Presynaptic density (PSD) size - shading around the presynaptic ',
            'Location (on spines, dendrites)', 'size of presyn compartment',
            'size of postsyn compartment', 'single synapse or dyad (or >)',
            'cleft thickness (how pronounced/obvious)', 'size of the vesicle cloud',
            'packing density', 'number of docked vesicles', 
            'mitochondria close by (<300nm from cleft)?'
        ]
        
        # Create a directory for saving plots
        plots_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create countplots for each feature
        for feature in categorical_features:
            if feature in df.columns:
                try:
                    plt.figure(figsize=(12, 6))
                    ax = sns.countplot(data=df, x=feature, palette='viridis')
                    ax.set_title(f'Distribution of {feature}')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Count')
                    
                    # Rotate x-axis labels if they are too long
                    if feature in ['Location (on spines, dendrites)', 'Presynaptic density (PSD) size - shading around the presynaptic ']:
                        plt.xticks(rotation=45, ha='right')
                    
                    # Adjust layout and save
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'{feature.replace(" ", "_").replace("(", "").replace(")", "").replace("?", "")}_distribution.png'))
                    plt.close()
                except Exception as e:
                    print(f"Error plotting feature {feature}: {e}")
                    plt.close()  # Make sure to close the figure if an error occurs
        
        print(f"Feature distribution plots saved to {plots_dir}")
    except Exception as e:
        print(f"Error in visualize_feature_distributions: {e}")
        traceback.print_exc()

def list_available_files():
    """List files in the current directory to help with debugging"""
    print("\nFiles in script directory:")
    for file in os.listdir(script_dir):
        print(f"- {file}")

def simple_analysis(df):
    """Perform simple analysis on the synapse data without external dependencies"""
    if df is None:
        print("Cannot perform simple analysis: No data provided")
        return
        
    try:
        print("\nSummary of synapse counts by bbox:")
        bbox_counts = df['bbox'].value_counts().sort_index()
        for bbox, count in bbox_counts.items():
            print(f"  Bbox {bbox}: {count} synapses")
        
        print("\nMost common feature values:")
        categorical_features = [
            'vesicle size', 'shape (roundness)', 
            'Location (on spines, dendrites)', 'size of presyn compartment',
            'single synapse or dyad (or >)', 'packing density'
        ]
        
        for feature in categorical_features:
            if feature in df.columns:
                value_counts = df[feature].value_counts()
                if len(value_counts) > 0:
                    most_common = value_counts.index[0]
                    percentage = (value_counts[most_common] / len(df)) * 100
                    print(f"  {feature}: {most_common} ({percentage:.1f}%)")
    
    except Exception as e:
        print(f"Error in simple_analysis: {e}")
        traceback.print_exc()

def visualize_bbox_compositions(df):
    """Visualize the composition of features within each bbox"""
    bboxes = sorted(df['bbox'].unique())
    
    # Create a directory for saving plots
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # For each bbox, show the distribution of key features
    key_features = ['vesicle size', 'Location (on spines, dendrites)', 'single synapse or dyad (or >)']
    
    for feature in key_features:
        if feature in df.columns:
            plt.figure(figsize=(14, 8))
            
            # Create a crosstab and visualize as a heatmap
            ct = pd.crosstab(df['bbox'], df[feature])
            sns.heatmap(ct, annot=True, cmap='viridis', fmt='d')
            
            plt.title(f'Distribution of {feature} across Bboxes')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'bbox_{feature.replace(" ", "_").replace("(", "").replace(")", "").replace("?", "")}_heatmap.png'))
            plt.close()
    
    print(f"Bbox composition plots saved to {plots_dir}")

def prepare_data_for_dimensionality_reduction(df):
    """Prepare categorical data for dimensionality reduction and clustering"""
    if not SKLEARN_AVAILABLE:
        print("Cannot perform dimensionality reduction: scikit-learn not available")
        return None
        
    try:
        # Select features for analysis
        features_for_analysis = [
            'vesicle size', 'shape (roundness)', 'Shading inside large vesicles',
            'Presynaptic density (PSD) size - shading around the presynaptic ',
            'Location (on spines, dendrites)', 'size of presyn compartment',
            'size of postsyn compartment', 'single synapse or dyad (or >)',
            'cleft thickness (how pronounced/obvious)', 'size of the vesicle cloud',
            'packing density', 'number of docked vesicles',
            'mitochondria close by (<300nm from cleft)?'
        ]
        
        # Check which features are available in the dataframe
        features_to_use = [f for f in features_for_analysis if f in df.columns]
        
        # Create a copy of the dataframe with only the selected features
        df_encoded = df[features_to_use].copy()
        
        # Handle missing values
        df_encoded = df_encoded.fillna('missing')
        
        # One-hot encode all categorical variables
        df_encoded_dummies = pd.get_dummies(df_encoded, drop_first=False)
        
        print(f"Data prepared for dimensionality reduction with {df_encoded_dummies.shape[1]} features after encoding")
        return df_encoded_dummies, features_to_use
    
    except Exception as e:
        print(f"Error preparing data for dimensionality reduction: {e}")
        traceback.print_exc()
        return None, None

def perform_dimensionality_reduction_and_clustering(df):
    """Perform dimensionality reduction (PCA, t-SNE, UMAP) and clustering on the data"""
    if df is None or not SKLEARN_AVAILABLE:
        print("Cannot perform dimensionality reduction: Data not available or scikit-learn missing")
        return df
        
    try:
        # Create directory for plots
        plots_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prepare data
        df_encoded_dummies, features_to_use = prepare_data_for_dimensionality_reduction(df)
        if df_encoded_dummies is None:
            return df
            
        # Scale the data for better dimensionality reduction
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_encoded_dummies)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)
        
        # Add PCA results to the original dataframe
        df['pca_x'] = pca_result[:, 0]
        df['pca_y'] = pca_result[:, 1]
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
        tsne_result = tsne.fit_transform(df_scaled)
        
        # Add t-SNE results to the original dataframe
        df['tsne_x'] = tsne_result[:, 0]
        df['tsne_y'] = tsne_result[:, 1]
        
        # Perform UMAP if available
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(random_state=42)
            umap_result = reducer.fit_transform(df_scaled)
            
            # Add UMAP results to the original dataframe
            df['umap_x'] = umap_result[:, 0]
            df['umap_y'] = umap_result[:, 1]
        
        # Perform K-means clustering
        # Determine optimal number of clusters using elbow method
        inertia = []
        K_range = range(2, min(10, len(df)))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_scaled)
            inertia.append(kmeans.inertia_)
        
        # Plot the elbow method results
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertia, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.savefig(os.path.join(plots_dir, 'elbow_method.png'))
        plt.close()
        
        # Choose a number of clusters (can be adjusted after viewing the elbow plot)
        n_clusters = min(4, len(df))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['kmeans_cluster'] = kmeans.fit_predict(df_scaled)
        
        # Also try DBSCAN for comparison
        dbscan = DBSCAN(eps=3, min_samples=2)
        df['dbscan_cluster'] = dbscan.fit_predict(df_scaled)
        
        # Create visualizations with both clustering methods
        plot_dimensionality_reduction(df, 'pca', plots_dir)
        plot_dimensionality_reduction(df, 'tsne', plots_dir)
        if UMAP_AVAILABLE:
            plot_dimensionality_reduction(df, 'umap', plots_dir)
            
        # Create cluster profile summaries
        create_cluster_profiles(df, features_to_use, 'kmeans_cluster', n_clusters, plots_dir)
        
        print("Dimensionality reduction and clustering completed successfully")
        return df
        
    except Exception as e:
        print(f"Error in dimensionality reduction and clustering: {e}")
        traceback.print_exc()
        return df

def plot_dimensionality_reduction(df, method, plots_dir):
    """Create plots for the dimensionality reduction methods with cluster coloring"""
    if method not in ['pca', 'tsne', 'umap']:
        print(f"Unknown dimensionality reduction method: {method}")
        return
        
    x_col = f"{method}_x"
    y_col = f"{method}_y"
    
    if x_col not in df.columns or y_col not in df.columns:
        print(f"{method.upper()} results not found in dataframe")
        return
        
    # Plot with K-means clusters
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot
    scatter = plt.scatter(df[x_col], df[y_col], c=df['kmeans_cluster'], 
                         cmap='viridis', s=70, alpha=0.8, edgecolors='w')
    
    # Add bbox information as text labels
    for bbox in df['bbox'].unique():
        bbox_df = df[df['bbox'] == bbox]
        centroid_x = bbox_df[x_col].mean()
        centroid_y = bbox_df[y_col].mean()
        plt.text(centroid_x, centroid_y, f"Bbox {bbox}", 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add a colorbar for cluster identification
    cbar = plt.colorbar(scatter)
    cbar.set_label('K-means Cluster')
    
    plt.title(f'{method.upper()} Visualization with K-means Clusters')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{method}_kmeans_clusters.png'), dpi=300)
    plt.close()
    
    # Plot with DBSCAN clusters
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with DBSCAN clusters
    scatter = plt.scatter(df[x_col], df[y_col], c=df['dbscan_cluster'], 
                         cmap='viridis', s=70, alpha=0.8, edgecolors='w')
    
    # Add bbox information
    for bbox in df['bbox'].unique():
        bbox_df = df[df['bbox'] == bbox]
        centroid_x = bbox_df[x_col].mean()
        centroid_y = bbox_df[y_col].mean()
        plt.text(centroid_x, centroid_y, f"Bbox {bbox}", 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('DBSCAN Cluster')
    
    plt.title(f'{method.upper()} Visualization with DBSCAN Clusters')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{method}_dbscan_clusters.png'), dpi=300)
    plt.close()
    
    # Plot with bbox coloring
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with bbox coloring
    scatter = plt.scatter(df[x_col], df[y_col], c=df['bbox'], 
                         cmap='tab10', s=70, alpha=0.8, edgecolors='w')
    
    # Add bbox labels
    for bbox in df['bbox'].unique():
        bbox_df = df[df['bbox'] == bbox]
        centroid_x = bbox_df[x_col].mean()
        centroid_y = bbox_df[y_col].mean()
        plt.text(centroid_x, centroid_y, f"Bbox {bbox}", 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Bbox Number')
    
    plt.title(f'{method.upper()} Visualization with Bbox Coloring')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{method}_bbox_coloring.png'), dpi=300)
    plt.close()

def create_cluster_profiles(df, features, cluster_col, n_clusters, plots_dir):
    """Create a summary of the most common attribute values in each cluster"""
    try:
        # Create a DataFrame to store cluster profiles
        cluster_profiles = pd.DataFrame()
        
        # For each feature, find the most common value in each cluster
        for feature in features:
            for cluster_id in range(n_clusters):
                cluster_data = df[df[cluster_col] == cluster_id]
                if len(cluster_data) > 0 and feature in cluster_data.columns:
                    mode_values = cluster_data[feature].mode()
                    if len(mode_values) > 0:
                        most_common = mode_values[0]
                        cluster_profiles.loc[feature, f'Cluster {cluster_id}'] = most_common
                    else:
                        cluster_profiles.loc[feature, f'Cluster {cluster_id}'] = 'NA'
        
        # Save cluster profiles to a CSV file
        profile_path = os.path.join(script_dir, 'cluster_profiles.csv')
        cluster_profiles.to_csv(profile_path)
        print(f"Cluster profiles saved to {profile_path}")
        
        # Create a heatmap of the cluster profiles
        plt.figure(figsize=(14, 10))
        
        # Create a numerical version of the profiles for the heatmap
        profiles_encoded = cluster_profiles.copy()
        for feature in profiles_encoded.index:
            feature_values = df[feature].dropna().unique()
            value_map = {val: i for i, val in enumerate(feature_values)}
            for col in profiles_encoded.columns:
                if profiles_encoded.loc[feature, col] in value_map:
                    profiles_encoded.loc[feature, col] = value_map[profiles_encoded.loc[feature, col]]
                else:
                    profiles_encoded.loc[feature, col] = np.nan
        
        # Create heatmap
        profiles_encoded = profiles_encoded.apply(pd.to_numeric, errors='coerce')
        sns.heatmap(profiles_encoded, cmap='viridis', annot=cluster_profiles.values, 
                   fmt='s', linewidths=0.5, cbar=False)
        plt.title('Cluster Profiles - Most Common Values')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cluster_profiles_heatmap.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating cluster profiles: {e}")
        traceback.print_exc()

def analyze_feature_combinations(df):
    """Analyze how features combine together across synapses"""
    if df is None:
        print("Cannot analyze feature combinations: No data provided")
        return
        
    try:
        # Create directory for plots
        plots_dir = os.path.join(script_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Select pairs of features to analyze
        feature_pairs = [
            ('vesicle size', 'packing density'),
            ('vesicle size', 'Location (on spines, dendrites)'),
            ('single synapse or dyad (or >)', 'size of presyn compartment'),
            ('size of presyn compartment', 'size of postsyn compartment')
        ]
        
        # Create contingency tables and visualize them
        for feature1, feature2 in feature_pairs:
            if feature1 in df.columns and feature2 in df.columns:
                # Create a crosstab
                ct = pd.crosstab(df[feature1], df[feature2])
                
                # Create a heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(ct, annot=True, cmap='viridis', fmt='d', cbar=True)
                plt.title(f'Relationship Between {feature1} and {feature2}')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'relationship_{feature1.replace(" ", "_")}_{feature2.replace(" ", "_")}.png'))
                plt.close()
                
                # Create a normalized heatmap (percentage)
                plt.figure(figsize=(10, 8))
                ct_norm = ct.div(ct.sum(axis=1), axis=0)
                sns.heatmap(ct_norm, annot=True, cmap='viridis', fmt='.2f', cbar=True)
                plt.title(f'Normalized Relationship Between {feature1} and {feature2}')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'normalized_relationship_{feature1.replace(" ", "_")}_{feature2.replace(" ", "_")}.png'))
                plt.close()
        
        print(f"Feature combination analysis completed and saved to {plots_dir}")
        
    except Exception as e:
        print(f"Error analyzing feature combinations: {e}")
        traceback.print_exc()

def main():
    print("Synapse Data Analysis Tool")
    print("-" * 40)
    
    # Check for the CSV file
    print(f"Looking for data file: {csv_path}")
    
    # List available files for debugging
    list_available_files()
    
    # Load the data
    print("\nLoading synapse data...")
    df = load_data()
    
    if df is not None:
        # Perform simple analysis that doesn't require external dependencies
        simple_analysis(df)
        
        # Generate feature distribution visualizations
        print("\nGenerating feature distribution visualizations...")
        visualize_feature_distributions(df)
        
        # Analyze feature combinations
        print("\nAnalyzing feature combinations...")
        analyze_feature_combinations(df)
        
        # Perform dimensionality reduction and clustering if possible
        if SKLEARN_AVAILABLE:
            print("\nPerforming dimensionality reduction and clustering...")
            df_with_clusters = perform_dimensionality_reduction_and_clustering(df)
            
            # Save the enhanced dataframe with dimensionality reduction results
            enhanced_df_path = os.path.join(script_dir, 'enhanced_df.csv')
            df_with_clusters.to_csv(enhanced_df_path, index=False)
            print(f"Enhanced dataframe saved to {enhanced_df_path}")
        else:
            print("\nSkipping dimensionality reduction and clustering - scikit-learn not available.")
            
        print("\nAnalysis complete.")
    else:
        print("Could not load data file. Analysis aborted.")

if __name__ == "__main__":
    main()
