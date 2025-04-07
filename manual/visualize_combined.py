import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA
import colorsys

# Import from main config
from synapse import config

def load_results():
    """
    Load the previously generated results
    """
    print("Loading results...")
    
    # Set up paths (same as in manual_feature_extraction.py)
    results_dir = 'manual/clustering_results'
    
    # Load manual clustering results
    manual_file = os.path.join(results_dir, "manual_clustered_samples.csv")
    if not os.path.exists(manual_file):
        raise FileNotFoundError(f"Manual clustering file not found: {manual_file}")
    manual_df = pd.read_csv(manual_file)
    print(f"Loaded manual clustering with {len(manual_df)} samples")
    
    # Load VGG features
    vgg_file = os.path.join(results_dir, "vgg_stage_specific_features.csv")
    if not os.path.exists(vgg_file):
        raise FileNotFoundError(f"VGG features file not found: {vgg_file}")
    vgg_df = pd.read_csv(vgg_file)
    print(f"Loaded VGG features with {len(vgg_df)} samples")
    
    # Load clustered features for additional info
    clustered_file = os.path.join(results_dir, "vgg_clustered_features.csv")
    if os.path.exists(clustered_file):
        clustered_df = pd.read_csv(clustered_file)
        print(f"Loaded clustered features with {len(clustered_df)} samples")
    else:
        clustered_df = None
        print("No clustered features file found")
    
    return manual_df, vgg_df, clustered_df

def create_feature_projections(manual_df, vgg_df, output_dir):
    """
    Create UMAP and PCA projections of VGG features colored by manual clusters
    """
    print("Creating feature projections with manual cluster coloring...")
    
    # Merge DataFrames
    merged_df = manual_df[['bbox_name', 'Var1', 'Manual_Cluster']].merge(
        vgg_df, on=['bbox_name', 'Var1']
    )
    
    # Get feature columns
    feature_cols = [col for col in vgg_df.columns if 'feat_' in col]
    print(f"Using {len(feature_cols)} feature columns for projections")
    
    # Extract features and standardize
    features = merged_df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    print("Applying UMAP...")
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(features_scaled)
    
    # Add UMAP results to DataFrame
    merged_df['UMAP1'] = umap_result[:, 0]
    merged_df['UMAP2'] = umap_result[:, 1]
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # Add PCA results to DataFrame
    merged_df['PCA1'] = pca_result[:, 0]
    merged_df['PCA2'] = pca_result[:, 1]
    
    # Save the merged DataFrame
    merged_df.to_csv(os.path.join(output_dir, "merged_projections.csv"), index=False)
    
    # Create colormap for manual clusters
    unique_clusters = sorted(merged_df['Manual_Cluster'].unique())
    n_clusters = len(unique_clusters)
    colors = [plt.cm.viridis(i/max(1, n_clusters-1)) for i in range(n_clusters)]
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Plot UMAP with manual cluster colors
    plt.figure(figsize=(12, 10))
    for cluster, group in merged_df.groupby('Manual_Cluster'):
        plt.scatter(
            group['UMAP1'], 
            group['UMAP2'],
            label=f"Manual Cluster {cluster}",
            color=cluster_colors[cluster],
            s=100,
            alpha=0.8
        )
    
    # Add synapse labels for better identification
    for i, row in merged_df.iterrows():
        label = row['Var1'].split('_')[-1]  # Just use the last part of the name
        plt.annotate(
            label,
            (row['UMAP1'], row['UMAP2']),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title("UMAP of VGG Features Colored by Manual Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vgg_umap_manual_colors.png"))
    
    # Plot PCA with manual cluster colors
    plt.figure(figsize=(12, 10))
    for cluster, group in merged_df.groupby('Manual_Cluster'):
        plt.scatter(
            group['PCA1'], 
            group['PCA2'],
            label=f"Manual Cluster {cluster}",
            color=cluster_colors[cluster],
            s=100,
            alpha=0.8
        )
    
    # Add synapse labels
    for i, row in merged_df.iterrows():
        label = row['Var1'].split('_')[-1]
        plt.annotate(
            label,
            (row['PCA1'], row['PCA2']),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title("PCA of VGG Features Colored by Manual Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vgg_pca_manual_colors.png"))
    
    print("Feature projections complete")
    return merged_df

def create_feature_heatmap(merged_df, output_dir):
    """
    Create a feature heatmap grouped by manual clusters
    """
    print("Creating feature heatmap grouped by manual clusters...")
    
    # Get feature columns
    feature_cols = [col for col in merged_df.columns if 'feat_' in col]
    
    # Sort by manual cluster
    sorted_df = merged_df.sort_values(by='Manual_Cluster')
    
    # Create a subset for the heatmap - use only features and cluster info
    heatmap_df = sorted_df[feature_cols].copy()
    
    # Standardize features for better visualization
    scaler = StandardScaler()
    heatmap_data = scaler.fit_transform(heatmap_df)
    heatmap_df = pd.DataFrame(heatmap_data, columns=feature_cols, index=heatmap_df.index)
    
    # Add cluster information for row colors
    cluster_info = sorted_df['Manual_Cluster'].values
    
    # Create row colors
    unique_clusters = sorted(np.unique(cluster_info))
    n_clusters = len(unique_clusters)
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Map cluster indices to colors
    cluster_to_idx = {cluster: i for i, cluster in enumerate(unique_clusters)}
    row_colors = [cluster_colors[cluster_to_idx[cluster]] for cluster in cluster_info]
    
    # Create the heatmap
    plt.figure(figsize=(16, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Ensure we don't have too many features for visualization
    if len(feature_cols) > 50:
        # Sample every nth feature to keep it manageable
        n = len(feature_cols) // 50 + 1
        sample_cols = feature_cols[::n]
        heatmap_df = heatmap_df[sample_cols]
        print(f"Sampled {len(sample_cols)} features for heatmap visualization")
    
    g = sns.clustermap(
        heatmap_df, 
        cmap=cmap,
        row_colors=row_colors,
        figsize=(16, 10),
        col_cluster=True,
        row_cluster=False,
        dendrogram_ratio=(.1, .2),
        cbar_pos=(0.02, .2, .03, .4)
    )
    
    # Add a legend for the row colors
    handles = [plt.Rectangle((0,0), 1, 1, color=cluster_colors[cluster_to_idx[cluster]]) for cluster in unique_clusters]
    plt.legend(
        handles, 
        [f"Cluster {cluster}" for cluster in unique_clusters],
        title="Manual Clusters",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper right'
    )
    
    plt.title("Feature Heatmap Grouped by Manual Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_heatmap.png"), dpi=300, bbox_inches='tight')
    
    print("Heatmap creation complete")

def analyze_feature_importance(merged_df, output_dir):
    """
    Analyze feature importance for distinguishing manual clusters
    """
    print("Analyzing feature importance...")
    
    # Get feature columns
    feature_cols = [col for col in merged_df.columns if 'feat_' in col]
    
    # Standardize features
    features = merged_df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_df = pd.DataFrame(features_scaled, columns=feature_cols)
    
    # Add cluster information
    features_df['Manual_Cluster'] = merged_df['Manual_Cluster'].values
    
    # Calculate feature importance using random forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    
    X = features_df[feature_cols]
    y = features_df['Manual_Cluster']
    
    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Get feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Calculate permutation importance for more robust results
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
    perm_importances = result.importances_mean
    perm_indices = np.argsort(perm_importances)[::-1]
    
    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': [feature_cols[i] for i in indices[:20]],
        'Importance': importances[indices[:20]]
    })
    
    # Create a DataFrame of permutation importances
    perm_importance_df = pd.DataFrame({
        'Feature': [feature_cols[i] for i in perm_indices[:20]],
        'Importance': perm_importances[perm_indices[:20]]
    })
    
    # Save feature importance DataFrames
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    perm_importance_df.to_csv(os.path.join(output_dir, "permutation_importance.csv"), index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(
        importance_df['Feature'][:20], 
        importance_df['Importance'][:20]
    )
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    
    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    plt.barh(
        perm_importance_df['Feature'][:20], 
        perm_importance_df['Importance'][:20]
    )
    plt.title("Top 20 Permutation Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "permutation_importance.png"))
    
    print("Feature importance analysis complete")

def main():
    # Set up directories
    output_dir = 'manual/visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    manual_df, vgg_df, clustered_df = load_results()
    
    # Create feature projections colored by manual clusters
    merged_df = create_feature_projections(manual_df, vgg_df, output_dir)
    
    # Create feature heatmap
    create_feature_heatmap(merged_df, output_dir)
    
    # Analyze feature importance
    analyze_feature_importance(merged_df, output_dir)
    
    print(f"All visualizations complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 