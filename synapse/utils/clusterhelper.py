import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Function to load features and perform clustering
def load_and_cluster_features(csv_filepath, n_clusters=5):
    features_df = pd.read_csv(csv_filepath)

    # Extract feature columns (assuming features start with 'feat_')
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    features = features_df[feature_cols].values

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_df['cluster'] = kmeans.fit_predict(features)

    return features_df, kmeans, feature_cols

def find_random_samples_in_clusters(features_df, feature_cols, n_samples=4):
    random_samples_per_cluster = {}

    # For each cluster, randomly select n_samples
    for cluster_id in np.unique(features_df['cluster']):
        cluster_samples = features_df[features_df['cluster'] == cluster_id]

        # Randomly select n_samples from the cluster
        if len(cluster_samples) >= n_samples:
            random_indices = np.random.choice(cluster_samples.index, n_samples, replace=False)
            random_samples = cluster_samples.loc[random_indices]
        else:
            # If there are fewer samples than n_samples, select all available samples
            random_samples = cluster_samples

        # Store the randomly selected samples for each cluster
        random_samples_per_cluster[cluster_id] = random_samples

    return random_samples_per_cluster

# Function to get the center slice of a 3D synapse sample
def get_center_slice(sample):
    """
    Extract the center slice from a 3D sample.
    Assumes the sample is a 3D numpy array.
    """
    # Get the center slice (middle slice in z-direction)
    center_slice = sample[sample.shape[0] // 2, :, :]
    return center_slice

# Function to plot 4 similar samples in a grid for each synapse
def plot_synapse_samples(dataset, closest_samples_indices, title='Synapse Samples'):
    """
    Plots 4 sample images of synapses in a grid layout.
    `dataset` is the dataset containing synapse 3D data.
    `closest_samples_indices` is a list of indices for the samples to plot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # 1 row, 4 columns

    for i, sample_idx in enumerate(closest_samples_indices):
        # Retrieve the synapse data (3D volume) from the dataset
        pixel_values, syn_info, bbox_name = dataset[sample_idx]  # Assuming dataset[index] gives 3D data

        # Get the center slice of the sample
        center_slice = get_center_slice(pixel_values)
        center_slice = center_slice.squeeze()
        # Plot the slice
        axes[i].imshow(center_slice, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}\n({syn_info["bbox_name"]})')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Function to find 4 closest samples from each cluster
def find_closest_samples_in_clusters(features_df, feature_cols, n_samples=4):
    closest_samples_per_cluster = {}

    # For each cluster, find the closest 4 samples based on feature similarity
    for cluster_id in np.unique(features_df['cluster']):
        cluster_samples = features_df[features_df['cluster'] == cluster_id]
        cluster_features = cluster_samples[feature_cols].values

        # Compute pairwise distances and select the closest pairs
        distances = pairwise_distances_argmin_min(cluster_features, cluster_features)

        # Select the closest samples (4 closest samples)
        closest_samples = []
        for i, sample_idx in enumerate(distances[0][:n_samples]):
            closest_samples.append(cluster_samples.iloc[sample_idx])

        # Store the closest samples for each cluster
        closest_samples_per_cluster[cluster_id] = closest_samples

    return closest_samples_per_cluster

# Function to reduce features to 2D or 3D using t-SNE
def apply_tsne(features_df, feature_cols, n_components=2):
    features = features_df[feature_cols].values
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

# Function to plot 2D and 3D t-SNE with plotly
def plot_tsne(features_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping):
    # 2D Plot colored by `bbox_name`
    fig_2d = px.scatter(
        features_df,
        x=tsne_results_2d[:, 0],
        y=tsne_results_2d[:, 1],
        color=features_df['bbox_name'],
        color_discrete_map=color_mapping,
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        title='2D t-SNE colored by bbox_name'
    )
    fig_2d.update_traces(marker=dict(size=4))  # Set the size of points to 2

    fig_2d.show()

    # 2D Plot colored by `cluster`
    fig_2d_cluster = px.scatter(
        features_df,
        x=tsne_results_2d[:, 0],
        y=tsne_results_2d[:, 1],
        color=kmeans.labels_.astype(str),  # Color by cluster number
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        title='2D t-SNE colored by cluster'
    )
    fig_2d_cluster.update_traces(marker=dict(size=4))  # Set the size of points to 2

    fig_2d_cluster.show()

# Function to save t-SNE plots using matplotlib
def save_tsne_plots(features_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping, output_dir):
    # 2D Plot colored by bbox_name - Matplotlib version
    plt.figure(figsize=(12, 10))
    bbox_groups = features_df.groupby('bbox_name')
    for bbox_name, group in bbox_groups:
        color = color_mapping.get(bbox_name, 'gray')
        indices = group.index
        plt.scatter(
            tsne_results_2d[indices, 0],
            tsne_results_2d[indices, 1],
            c=color,
            label=bbox_name,
            alpha=0.7,
            s=50
        )
    plt.title('2D t-SNE colored by bbox_name')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_2d_bbox.png"), dpi=300)
    plt.close()

    # 2D Plot colored by cluster - Matplotlib version
    plt.figure(figsize=(12, 10))
    for cluster_id in range(kmeans.n_clusters):
        indices = np.where(kmeans.labels_ == cluster_id)[0]
        plt.scatter(
            tsne_results_2d[indices, 0],
            tsne_results_2d[indices, 1],
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=50
        )
    plt.title('2D t-SNE colored by cluster')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_2d_cluster.png"), dpi=300)
    plt.close()

    # 3D Plot - Save as static image instead of HTML
    if tsne_results_3d is not None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in range(kmeans.n_clusters):
            indices = np.where(kmeans.labels_ == cluster_id)[0]
            ax.scatter(
                tsne_results_3d[indices, 0],
                tsne_results_3d[indices, 1],
                tsne_results_3d[indices, 2],
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=50
            )
        
        ax.set_title('3D t-SNE')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_3d.png"), dpi=300)
        plt.close()

# Function to save cluster samples as images
def save_cluster_samples(dataset, closest_samples_per_cluster, output_dir):
    for cluster_id, samples in closest_samples_per_cluster.items():
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        for i, sample in enumerate(samples):
            pixel_values, syn_info, bbox_name = dataset[sample.name]
            center_slice = pixel_values[pixel_values.shape[0] // 2, :, :].squeeze()
            axes[i].imshow(center_slice, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}\n({syn_info["bbox_name"]})')
        plt.suptitle(f'Cluster {cluster_id} Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_samples.png'))
        plt.close() 