import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

def load_and_cluster_features(csv_filepath, n_clusters=5):
    features_df = pd.read_csv(csv_filepath)

    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    features = features_df[feature_cols].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_df['cluster'] = kmeans.fit_predict(features)

    return features_df, kmeans, feature_cols

def find_random_samples_in_clusters(features_df, feature_cols, n_samples=4):
    random_samples_per_cluster = {}

    for cluster_id in np.unique(features_df['cluster']):
        cluster_samples = features_df[features_df['cluster'] == cluster_id]

        if len(cluster_samples) >= n_samples:
            random_indices = np.random.choice(cluster_samples.index, n_samples, replace=False)
            random_samples = cluster_samples.loc[random_indices]
        else:
            random_samples = cluster_samples

        random_samples_per_cluster[cluster_id] = random_samples

    return random_samples_per_cluster

def get_center_slice(sample):
    center_slice = sample[sample.shape[0] // 2, :, :]
    return center_slice

def visualize_slice(ax, slice_data, title=None, consistent_gray=True):
    """
    Consistently visualize a 2D slice with controlled normalization.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to plot on
        slice_data (numpy.ndarray): 2D slice data
        title (str, optional): Title for the plot
        consistent_gray (bool): Whether to enforce consistent gray levels
        
    Returns:
        matplotlib.image.AxesImage: The displayed image
    """
    # Ensure the slice is 2D
    if len(slice_data.shape) > 2:
        slice_data = slice_data.squeeze()
    
    # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
    # which can change the appearance of gray overlays
    if consistent_gray:
        im = ax.imshow(slice_data, cmap='gray', vmin=0, vmax=1)
    else:
        # Fall back to matplotlib's auto-scaling for special cases
        im = ax.imshow(slice_data, cmap='gray')
    
    ax.axis('off')
    if title:
        ax.set_title(title)
    
    return im

def plot_synapse_samples(dataset, closest_samples_indices, title='Synapse Samples'):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    for i, sample_idx in enumerate(closest_samples_indices):
        pixel_values, syn_info, bbox_name = dataset[sample_idx]
        center_slice = get_center_slice(pixel_values)
        
        # Use the consistent visualization function
        visualize_slice(
            axes[i], 
            center_slice, 
            title=f'Sample {i+1}\n({syn_info["bbox_name"]})'
        )

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def find_closest_samples_in_clusters(features_df, feature_cols, n_samples=4):
    closest_samples_per_cluster = {}

    for cluster_id in np.unique(features_df['cluster']):
        cluster_samples = features_df[features_df['cluster'] == cluster_id]
        cluster_features = cluster_samples[feature_cols].values

        distances = pairwise_distances_argmin_min(cluster_features, cluster_features)

        closest_samples = []
        for i, sample_idx in enumerate(distances[0][:n_samples]):
            closest_samples.append(cluster_samples.iloc[sample_idx])

        closest_samples_per_cluster[cluster_id] = closest_samples

    return closest_samples_per_cluster

def apply_tsne(features_df, feature_cols, n_components=2):
    features = features_df[feature_cols].values
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

def plot_tsne(features_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping):
    fig_2d = px.scatter(
        features_df,
        x=tsne_results_2d[:, 0],
        y=tsne_results_2d[:, 1],
        color=features_df['bbox_name'],
        color_discrete_map=color_mapping,
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        title='2D t-SNE colored by bbox_name'
    )
    fig_2d.update_traces(marker=dict(size=4))

    fig_2d.show()

    fig_2d_cluster = px.scatter(
        features_df,
        x=tsne_results_2d[:, 0],
        y=tsne_results_2d[:, 1],
        color=kmeans.labels_.astype(str),
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        title='2D t-SNE colored by cluster'
    )
    fig_2d_cluster.update_traces(marker=dict(size=4))

    fig_2d_cluster.show()

def save_tsne_plots(features_df, tsne_results_2d, tsne_results_3d, kmeans, color_mapping, output_dir):
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

def save_cluster_samples(dataset, closest_samples_per_cluster, output_dir):
    for cluster_id, samples in closest_samples_per_cluster.items():
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        for i, (idx, _) in enumerate(samples.iterrows()):
            try:
                pixel_values, syn_info, bbox_name = dataset[idx]
                center_slice = pixel_values[pixel_values.shape[0] // 2, :, :]
                
                # Use the consistent visualization function
                visualize_slice(
                    axes[i], 
                    center_slice, 
                    title=f'Sample {i+1}\n({syn_info["bbox_name"]})'
                )
            except Exception as e:
                print(f"Error processing sample at index {idx}: {e}")
                axes[i].axis('off')
                axes[i].set_title(f'Sample {i+1}\n(Error)')
                
        plt.suptitle(f'Cluster {cluster_id} Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_samples.png'))
        plt.close() 