import os
import pandas as pd
import numpy as np
import gower
import torch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
import umap

# Import from newdl module
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset
from synapse import (
    Vgg3D, 
    load_model_from_checkpoint,
    config
)

# Import VGG3DStageExtractor for feature extraction
from vgg3d_stage_extractor import VGG3DStageExtractor

# Import clustering utilities 
from synapse.clustering import (
    load_and_cluster_features,
    find_random_samples_in_clusters
)

def load_manual_annotations():
    """
    Load the manual annotations from the annotation.csv file
    """
    print("Loading manual annotations...")
    
    # Load annotation.csv file
    annotation_df = pd.read_csv('manual/annotation.csv', header=None)
    
    # Extract bbox (first row) and synapse name (second row)
    bbox_names = annotation_df.iloc[0].values[1:]  # Skip the first column (contains "bbox:")
    synapse_names = annotation_df.iloc[1].values[1:]  # Skip the first column (contains "synapse:")
    
    # Create a list of valid synapses (where both bbox and synapse names are strings)
    synapses = []
    valid_indices = []  # Keep track of valid indices
    
    for i in range(len(bbox_names)):
        if pd.notna(bbox_names[i]) and pd.notna(synapse_names[i]):
            if isinstance(bbox_names[i], str) and isinstance(synapse_names[i], str):
                synapses.append({
                    'bbox_name': f"bbox{bbox_names[i]}",
                    'Var1': synapse_names[i]
                })
                valid_indices.append(i)
    
    # Features start from the third row
    feature_names = []
    features_data = []
    
    for row_idx in range(2, len(annotation_df)):
        row = annotation_df.iloc[row_idx]
        if pd.notna(row[0]):  # Check if feature name exists
            feature_name = row[0]
            if isinstance(feature_name, str) and not feature_name.strip().startswith("For further information"):
                feature_names.append(feature_name)
                # Only take feature values for valid synapse indices
                feature_values = [row.values[i + 1] for i in valid_indices]  # +1 because we skip first column
                features_data.append(feature_values)
    
    if not synapses or not feature_names:
        raise ValueError("No valid synapses or features found in the annotation file")
    
    # Create DataFrame with synapse information
    synapse_df = pd.DataFrame(synapses)
    
    # Create DataFrame with features
    features_df = pd.DataFrame(features_data).transpose()
    features_df.columns = feature_names
    
    print(f"Found {len(synapse_df)} valid synapses")
    print(f"Found {len(feature_names)} features")
    print(f"Feature matrix shape: {features_df.shape}")
    
    # Verify dimensions match
    if len(synapse_df) != len(features_df):
        raise ValueError(f"Mismatch between number of synapses ({len(synapse_df)}) and feature rows ({len(features_df)})")
    
    # Combine synapse info with features
    combined_df = pd.concat([synapse_df, features_df], axis=1)
    
    # Handle missing values
    combined_df = combined_df.replace('N/A', np.nan)
    
    print(f"Successfully loaded {len(combined_df)} manually annotated synapses with {len(feature_names)} features")
    return combined_df

def perform_manual_clustering(manual_df):
    """
    Perform clustering on the manual annotations, similar to manual.py
    """
    print("Performing manual clustering...")
    
    # Read metadata for feature types
    metadata = pd.read_csv('manual/metadata.csv', delimiter=";", header=None)
    feature_names = metadata.iloc[:, 0].dropna().tolist()
    feature_types = metadata.iloc[:, 1].dropna().tolist()
    categories_list = metadata.iloc[:, 2:].apply(lambda x: x.dropna().tolist(), axis=1).tolist()
    
    # Separate features by type
    nominal_features = [feature_names[i] for i, ftype in enumerate(feature_types) if ftype == "nominal"]
    ordinal_features = [feature_names[i] for i, ftype in enumerate(feature_types) if ftype == "ordinal"]
    
    # Create a dictionary of ordinal feature name -> list of categories
    ordinal_categories = {
        feature_names[i]: categories_list[i]
        for i, ftype in enumerate(feature_types)
        if ftype == "ordinal"
    }
    
    # Select only columns that exist in our data
    available_nominal = [f for f in nominal_features if f in manual_df.columns]
    available_ordinal = [f for f in ordinal_features if f in manual_df.columns]
    
    # Encode nominal features
    encoder_nominal = OneHotEncoder(sparse_output=False, drop="first")
    df_nominal = pd.DataFrame(index=manual_df.index)
    
    for feature in available_nominal:
        if not manual_df[feature].isnull().all():
            encoded = encoder_nominal.fit_transform(manual_df[[feature]])
            col_names = encoder_nominal.get_feature_names_out([feature])
            df_nominal = pd.concat([df_nominal, pd.DataFrame(encoded, index=manual_df.index, columns=col_names)], axis=1)
    
    # Encode ordinal features
    df_ordinal = pd.DataFrame(index=manual_df.index)
    if available_ordinal:
        # Filter ordinal_categories to include only available features
        filtered_categories = [ordinal_categories[f] for f in available_ordinal]
        
        # Replace NaN values with the most common value for each feature
        manual_ord_df = manual_df[available_ordinal].copy()
        for feat in available_ordinal:
            most_common = manual_ord_df[feat].mode()[0]
            manual_ord_df[feat] = manual_ord_df[feat].fillna(most_common)
        
        encoder_ordinal = OrdinalEncoder(categories=filtered_categories)
        encoded = encoder_ordinal.fit_transform(manual_ord_df)
        df_ordinal = pd.DataFrame(encoded, index=manual_df.index, columns=available_ordinal)
    
    # Combine encoded data
    df_encoded = pd.concat([df_nominal, df_ordinal], axis=1)
    
    # Compute Gower distance
    distance_matrix = gower.gower_matrix(df_encoded)
    
    # Hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Create dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=manual_df.index, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram (Manual Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "manual_clustering_dendrogram.png"))
    
    # Assign clusters (default 2 clusters)
    num_clusters = 2
    clusters = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
    df_encoded["Cluster"] = clusters
    
    # Combine with original data for analysis
    manual_df["Manual_Cluster"] = clusters
    
    # Save results
    manual_df.to_csv(os.path.join(config.clustering_output_dir, "manual_clustered_samples.csv"), index=False)
    
    # Create PCA visualization
    scaler = StandardScaler()
    df_encoded.columns = df_encoded.columns.astype(str)
    X_scaled = scaler.fit_transform(df_encoded.drop(columns=["Cluster"]))  # Exclude cluster for PCA
    
    # Apply PCA
    n_components = min(4, X_scaled.shape[1])  # Use at most 4 components, or however many features we have
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA components
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(pca_result, columns=pca_columns, index=df_encoded.index)
    df_pca["Cluster"] = df_encoded["Cluster"]
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="viridis", s=100, edgecolor="k")
    
    # Add synapse labels
    for i, label in enumerate(manual_df["Var1"]):
        plt.annotate(
            label.split("_")[-1],  # Use the last part of the synapse name
            (df_pca.iloc[i]["PC1"], df_pca.iloc[i]["PC2"]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title("PCA Visualization of Manual Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "manual_clusters_pca.png"))
    
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    
    # Create DataFrame with UMAP components
    df_umap = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"], index=df_encoded.index)
    df_umap["Cluster"] = df_encoded["Cluster"]
    
    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Cluster", palette="viridis", s=100, edgecolor="k")
    
    # Add synapse labels
    for i, label in enumerate(manual_df["Var1"]):
        plt.annotate(
            label.split("_")[-1],  # Use the last part of the synapse name
            (df_umap.iloc[i]["UMAP1"], df_umap.iloc[i]["UMAP2"]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title("UMAP Visualization of Manual Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "manual_clusters_umap.png"))
    
    print(f"Manual clustering completed with {num_clusters} clusters")
    return manual_df

def load_model():
    """
    Load the VGG3D model from checkpoint
    """
    print("Loading VGG3D model...")
    
    checkpoint_path = 'hemibrain_production.checkpoint'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)
    
    return model

def find_synapses_in_dataloader(manual_df, config):
    """
    Find the synapses in the dataloader based on bbox_name and Var1
    """
    print("Finding synapses in dataloader...")
    
    # Create a data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    vol_data_dict = {}
    bbox_names = set(manual_df['bbox_name'])
    
    for bbox_name in tqdm(bbox_names, desc="Loading volumes"):
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
    
    # Load synapse metadata from excel files
    syn_df = pd.concat([
        pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
        for bbox in bbox_names if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
    ])
    
    # Filter to only include synapses in our manual annotations
    filtered_syn_df = syn_df[syn_df.apply(lambda row: 
        (row['bbox_name'] in manual_df['bbox_name'].values) and 
        (row['Var1'] in manual_df['Var1'].values), axis=1)]
    
    print(f"Found {len(filtered_syn_df)} matching synapses in dataloader out of {len(manual_df)} manual synapses")
    
    return vol_data_dict, filtered_syn_df

def extract_stage_specific_features(model, vol_data_dict, syn_df, layer_num=20, preprocessing='intelligent_cropping', preprocessing_weight=0.7):
    """
    Extract features from a specific layer using the VGG3DStageExtractor
    """
    print(f"Extracting stage-specific features from layer {layer_num} with preprocessing={preprocessing}, weight={preprocessing_weight}...")
    
    # Create the stage extractor
    extractor = VGG3DStageExtractor(model)
    
    # Print information about the model stages
    stage_info = extractor.get_stage_info()
    for stage_num, info in stage_info.items():
        start_idx, end_idx = info['range']
        if start_idx <= layer_num <= end_idx:
            print(f"Layer {layer_num} is in Stage {stage_num}")
            break
    
    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    
    # Create dataset with intelligent cropping parameters
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        smart_crop=preprocessing == 'intelligent_cropping',
        presynapse_weight=preprocessing_weight,
        normalize_presynapse_size=True
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Extract features for each synapse
    features = []
    metadata = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            # Get sample
            sample = dataset[i]
            if sample is None:
                continue
                
            pixels, info, name = sample
            
            # Add batch dimension and move to device
            inputs = pixels.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            
            # Extract features from specified layer
            batch_features = extractor.extract_layer(layer_num, inputs)
            
            # Global average pooling to get a feature vector
            batch_size = batch_features.shape[0]
            num_channels = batch_features.shape[1]
            
            # Reshape to (batch_size, channels, -1) for easier processing
            batch_features_reshaped = batch_features.reshape(batch_size, num_channels, -1)
            
            # Global average pooling across spatial dimensions
            pooled_features = torch.mean(batch_features_reshaped, dim=2)
            
            # Convert to numpy
            features_np = pooled_features.cpu().numpy()
            
            features.append(features_np)
            metadata.append((name, info))
    
    if not features:
        raise ValueError("No features were extracted. Check your dataset and model.")
    
    # Combine features and metadata
    features = np.concatenate(features, axis=0)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame([
        {"bbox": name, **info.to_dict()}
        for name, info in metadata
    ])
    
    # Create feature DataFrame
    feature_columns = [f'layer{layer_num}_feat_{i+1}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)
    
    # Combine metadata and features
    combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    # Save features
    combined_df.to_csv(os.path.join(config.clustering_output_dir, "vgg_stage_specific_features.csv"), index=False)
    
    print(f"Extracted VGG stage-specific features for {len(combined_df)} synapses with {len(feature_columns)} features")
    return combined_df

def cluster_and_visualize_features(features_df, n_clusters=2):
    """
    Cluster the features and visualize with UMAP and PCA
    """
    print(f"Clustering and visualizing features using {n_clusters} clusters...")
    
    # Determine feature columns
    if any(col.startswith('layer') for col in features_df.columns):
        # Stage-specific features
        feature_cols = [col for col in features_df.columns if 'feat_' in col]
    else:
        # Standard features
        feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    
    print(f"Using {len(feature_cols)} feature columns for clustering")
    
    # Extract the features for clustering
    features = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to the DataFrame
    features_df['cluster'] = cluster_labels
    
    # Save clustered features
    features_df.to_csv(os.path.join(config.clustering_output_dir, "vgg_clustered_features.csv"), index=False)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # Add PCA results to the DataFrame
    features_df['PCA1'] = pca_result[:, 0]
    features_df['PCA2'] = pca_result[:, 1]
    
    # Plot PCA
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_df['PCA1'], 
        features_df['PCA2'], 
        c=features_df['cluster'], 
        cmap='viridis', 
        s=100, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title("PCA Visualization of VGG Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "vgg_clusters_pca.png"))
    
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(features_scaled)
    
    # Add UMAP results to the DataFrame
    features_df['UMAP1'] = umap_result[:, 0]
    features_df['UMAP2'] = umap_result[:, 1]
    
    # Plot UMAP
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_df['UMAP1'], 
        features_df['UMAP2'], 
        c=features_df['cluster'], 
        cmap='viridis', 
        s=100, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title("UMAP Visualization of VGG Features")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "vgg_clusters_umap.png"))
    
    # Also create plots colored by bbox
    bbox_colors = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    # Function to get color from bbox name
    def get_color(bbox):
        return bbox_colors.get(bbox, '#000000')
    
    # Add colors to DataFrame
    features_df['color'] = features_df['bbox_name'].apply(get_color)
    
    # Plot PCA with bbox colors
    plt.figure(figsize=(12, 10))
    for bbox, group in features_df.groupby('bbox_name'):
        plt.scatter(
            group['PCA1'], 
            group['PCA2'], 
            label=bbox,
            color=bbox_colors.get(bbox, '#000000'), 
            s=100, 
            alpha=0.8
        )
    plt.title("PCA Visualization of VGG Features by Bounding Box")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "vgg_pca_by_bbox.png"))
    
    # Plot UMAP with bbox colors
    plt.figure(figsize=(12, 10))
    for bbox, group in features_df.groupby('bbox_name'):
        plt.scatter(
            group['UMAP1'], 
            group['UMAP2'], 
            label=bbox,
            color=bbox_colors.get(bbox, '#000000'), 
            s=100, 
            alpha=0.8
        )
    plt.title("UMAP Visualization of VGG Features by Bounding Box")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "vgg_umap_by_bbox.png"))
    
    print("Clustering and visualization completed")
    return features_df

def compare_clusters(manual_df, vgg_df):
    """
    Compare the manual clusters with the VGG-based clusters
    """
    print("Comparing manual and VGG-based clusters...")
    
    # Merge the dataframes on bbox_name and Var1
    merged_df = manual_df.merge(
        vgg_df[['bbox_name', 'Var1', 'cluster', 'UMAP1', 'UMAP2']], 
        on=['bbox_name', 'Var1'],
        suffixes=('', '_vgg')
    )
    
    # Rename VGG cluster column
    merged_df = merged_df.rename(columns={'cluster': 'VGG_Cluster'})
    
    # Compute agreement metrics
    total_synapses = len(merged_df)
    
    # Create a cross-tabulation of manual vs VGG clusters
    contingency_table = pd.crosstab(
        merged_df['Manual_Cluster'], 
        merged_df['VGG_Cluster'],
        rownames=['Manual Cluster'],
        colnames=['VGG Cluster']
    )
    
    # Compute percentage agreement
    # Find the best mapping of VGG clusters to manual clusters
    best_agreement = 0
    for i in range(1, 3):  # For manual cluster 1
        for j in range(0, 2):  # For VGG cluster 0 or 1
            mapping = {i: j, 3-i: 1-j}  # Map 1->j and 2->(1-j)
            
            # Count agreements
            agreement_count = sum(merged_df['VGG_Cluster'] == merged_df['Manual_Cluster'].map(mapping))
            agreement_pct = (agreement_count / total_synapses) * 100
            
            if agreement_pct > best_agreement:
                best_agreement = agreement_pct
                best_mapping = mapping
    
    print(f"Contingency table of manual vs. VGG clusters:\n{contingency_table}")
    print(f"Best cluster agreement: {best_agreement:.1f}% with mapping {best_mapping}")
    
    # Save the merged data
    merged_df.to_csv(os.path.join(config.clustering_output_dir, "cluster_comparison.csv"), index=False)
    
    # Plot comparison using UMAP coordinates
    plt.figure(figsize=(12, 10))
    
    # Create a combined categorical variable for manual+VGG clusters
    merged_df['combined_cluster'] = merged_df['Manual_Cluster'].astype(str) + '-' + merged_df['VGG_Cluster'].astype(str)
    
    # Plot using UMAP coordinates from VGG
    for cluster, group in merged_df.groupby('combined_cluster'):
        plt.scatter(
            group['UMAP1'], 
            group['UMAP2'],
            label=f"Manual-VGG: {cluster}",
            s=100,
            alpha=0.8
        )
    
    plt.title("Comparison of Manual and VGG-based Clusters (UMAP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.clustering_output_dir, "cluster_comparison_umap.png"))
    
    return merged_df

def main():
    # Parse config from synapse/utils/config.py
    config.parse_args()
    
    # Override specific settings
    config.extraction_method = "stage_specific"
    config.layer_num = 20
    config.preprocessing = 'intelligent_cropping'
    config.preprocessing_weights = 0.7
    
    # Set up output directories in manual folder
    config.csv_output_dir = 'manual/csv_outputs'
    config.clustering_output_dir = 'manual/clustering_results'
    config.save_gifs_dir = 'manual/gifs'
    
    # Create necessary directories
    os.makedirs(config.csv_output_dir, exist_ok=True)
    os.makedirs(config.save_gifs_dir, exist_ok=True)
    os.makedirs(config.clustering_output_dir, exist_ok=True)
    
    # Step 1: Load manual annotations
    manual_df = load_manual_annotations()
    
    # Step 2: Perform manual clustering
    manual_df = perform_manual_clustering(manual_df)
    
    # Step 3: Load the VGG3D model
    model = load_model()
    
    # Step 4: Find synapses in dataloader
    vol_data_dict, syn_df = find_synapses_in_dataloader(manual_df, config)
    
    # Step 5: Extract VGG features using stage-specific extraction
    vgg_features_df = extract_stage_specific_features(
        model, 
        vol_data_dict, 
        syn_df, 
        layer_num=config.layer_num,
        preprocessing=config.preprocessing,
        preprocessing_weight=config.preprocessing_weights
    )
    
    # Step 6: Cluster VGG features and visualize with UMAP and PCA
    vgg_clustered_df = cluster_and_visualize_features(vgg_features_df, n_clusters=2)
    
    # Step 7: Compare manual and VGG-based clusters
    comparison_df = compare_clusters(manual_df, vgg_clustered_df)
    
    print(f"Analysis complete. Results saved to {config.clustering_output_dir}")

if __name__ == "__main__":
    main() 