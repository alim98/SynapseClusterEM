"""
Vesicle Size Visualization Module

This module provides functions to visualize vesicle cloud sizes in UMAP plots
and analyze the relationship between vesicle sizes and clusters.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
from scipy import stats
from scipy.stats import gaussian_kde
from umap import UMAP
import statsmodels.api as sm
from skimage import measure  # Import for component analysis


def get_closest_component_mask(mask, z_start, z_end, y_start, y_end, x_start, x_end, center_coord):
    """
    Find the connected component closest to the center coordinate.
    
    Args:
        mask: Binary mask of the target structure
        z_start, z_end, y_start, y_end, x_start, x_end: Bounds of the subvolume
        center_coord: (x, y, z) center coordinate
        
    Returns:
        Binary mask of the closest component
    """
    # Extract the subvolume
    subvol_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # If mask is empty, return it as is
    if not np.any(subvol_mask):
        return subvol_mask
    
    # Find all connected components
    labeled_mask, num_components = measure.label(subvol_mask, return_num=True, connectivity=3)
    
    # If only one component, return it
    if num_components <= 1:
        return subvol_mask
    
    # Convert center coordinates to subvolume coordinates
    cx, cy, cz = center_coord
    sub_cx = cx - x_start
    sub_cy = cy - y_start
    sub_cz = cz - z_start
    
    # Find the component closest to the center
    min_dist = float('inf')
    closest_label = 0
    
    # Get coordinates of all points in each component
    for label in range(1, num_components+1):
        component_coords = np.argwhere(labeled_mask == label)
        if len(component_coords) == 0:
            continue
        
        # Calculate distance to center for each point
        distances = np.sqrt(
            (component_coords[:, 0] - sub_cz)**2 + 
            (component_coords[:, 1] - sub_cy)**2 + 
            (component_coords[:, 2] - sub_cx)**2
        )
        min_component_dist = np.min(distances)
        
        # Update closest component
        if min_component_dist < min_dist:
            min_dist = min_component_dist
            closest_label = label
    
    # Create a mask with only the closest component
    closest_component_mask = np.zeros_like(subvol_mask)
    if closest_label > 0:
        closest_component_mask = (labeled_mask == closest_label)
    
    return closest_component_mask


def get_vesicle_label(bbox_name):
    """
    Determine vesicle label based on bbox name.
    
    Args:
        bbox_name: Name of the bounding box
        
    Returns:
        int: Vesicle label
    """
    bbox_num = bbox_name.replace("bbox", "").strip()
    if bbox_num in {'2', '5'}:
        return 3
    elif bbox_num == '7':
        return 2
    elif bbox_num == '4':
        return 2
    elif bbox_num == '3':
        return 7
    else:  # For bbox1, 6, etc.
        return 6


def calculate_vesicle_cloud_size(row, vol_data_dict, subvol_size):
    """
    Calculate the vesicle cloud mask size for a given synapse row.
    
    Args:
        row: Row from the synapse dataframe
        vol_data_dict: Dictionary containing volume data
        subvol_size: Subvolume size
        
    Returns:
        float: Vesicle cloud size as percentage of total 80×80×80 raw image
    """
    bbox_name = row['bbox_name']
    if bbox_name not in vol_data_dict:
        return 0  # Handle missing data

    add_mask_vol = vol_data_dict[bbox_name][2]  # Get vesicle segmentation volume
    vesicle_label = get_vesicle_label(bbox_name)

    # Extract coordinates from the dataframe (x, y, z)
    cx, cy, cz = (
        int(row['central_coord_1']),
        int(row['central_coord_2']),
        int(row['central_coord_3'])
    )

    # Calculate subvolume bounds
    half_size = subvol_size // 2
    x_start = max(cx - half_size, 0)
    x_end = min(cx + half_size, add_mask_vol.shape[2])
    y_start = max(cy - half_size, 0)
    y_end = min(cy + half_size, add_mask_vol.shape[1])
    z_start = max(cz - half_size, 0)
    z_end = min(cz + half_size, add_mask_vol.shape[0])

    # Generate full vesicle mask and find closest component
    vesicle_full_mask = (add_mask_vol == vesicle_label)
    
    vesicle_mask = get_closest_component_mask(
        vesicle_full_mask,
        z_start, z_end,
        y_start, y_end,
        x_start, x_end,
        (cx, cy, cz)
    )

    # Count total vesicle pixels
    total_vesicle_pixels = np.sum(vesicle_mask)
    
    # Calculate percentage based on the standard 80×80×80 volume size
    # regardless of actual subvolume boundaries
    standard_volume_size = 80 * 80 * 80
    vesicle_size_percent = (total_vesicle_pixels / standard_volume_size) * 100
    
    return vesicle_size_percent  # Return as percentage of standard 80×80×80 volume


def compute_vesicle_cloud_sizes(syn_df, vol_data_dict, args, output_dir):
    """
    Compute vesicle cloud size for each synapse and save to CSV.
    
    Args:
        syn_df: Synapse dataframe
        vol_data_dict: Dictionary containing volume data
        args: Arguments containing configuration
        output_dir: Directory to save results
        
    Returns:
        pandas.DataFrame: Synapse dataframe with vesicle cloud sizes (as percentage of 80×80×80 raw image)
    """
    print("Calculating vesicle cloud sizes (as % of 80×80×80 raw image)...")
    syn_df['vesicle_cloud_size'] = syn_df.apply(
        lambda row: calculate_vesicle_cloud_size(row, vol_data_dict, args.subvol_size),
        axis=1
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each bbox's dataframe with the new column
    for bbox in args.bbox_name:
        bbox_df = syn_df[syn_df['bbox_name'] == bbox]
        if not bbox_df.empty:
            output_path = os.path.join(output_dir, f"{bbox}.csv")
            bbox_df.to_csv(output_path, index=False)
            print(f"Saved {output_path} with vesicle cloud sizes (as % of 80×80×80 raw image).")
            
    return syn_df


def create_umap_with_vesicle_sizes(features_df, vesicle_df, output_dir):
    """
    Create UMAP visualization with vesicle sizes.
    
    Args:
        features_df: Features dataframe with cluster information
        vesicle_df: Dataframe with vesicle cloud sizes
        output_dir: Directory to save results
        
    Returns:
        tuple: (merged_df, umap_fig) The merged dataframe and the UMAP figure
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge datasets using composite key
    merged_df = pd.merge(
        features_df,
        vesicle_df[['bbox_name', 'central_coord_1', 'central_coord_2',
                    'central_coord_3', 'vesicle_cloud_size']],
        on=['bbox_name', 'central_coord_1', 'central_coord_2', 'central_coord_3'],
        how='inner'
    )

    # Create meaningful size categories
    merged_df['size_category'] = pd.qcut(merged_df['vesicle_cloud_size'],
                                        q=4,
                                        labels=['Small', 'Medium', 'Large', 'X-Large'])
    
    # Convert cluster to string to ensure distinct colors in the visualization
    merged_df['cluster_str'] = 'Cluster ' + merged_df['cluster'].astype(str)

    # Compute UMAP embedding
    print("Computing UMAP embedding for visualization...")
    umap_3d = UMAP(n_components=3, random_state=42)
    features = merged_df.filter(regex='feat_').values
    projection_3d = umap_3d.fit_transform(features)

    # Create 3D scatter plot with cluster coloring and vesicle sizes
    fig_umap_by_cluster = px.scatter_3d(
        merged_df,
        x=projection_3d[:,0],
        y=projection_3d[:,1],
        z=projection_3d[:,2],
        color='cluster_str',  # Use cluster_str to ensure categorical coloring
        size='vesicle_cloud_size',
        hover_data=['size_category', 'bbox_name', 'cluster'],
        title="3D Topological Manifold with Cluster Coloring & Vesicle Size (% of 80×80×80 Volume)",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold  # Use a qualitative color scale
    )
    fig_umap_by_cluster.update_layout(width=1200, height=800)
    
    # Update hover template to show percentages
    fig_umap_by_cluster.update_traces(
        hovertemplate='<b>Cluster:</b> %{customdata[2]}<br>' +
                      '<b>Size Category:</b> %{customdata[0]}<br>' +
                      '<b>Bbox:</b> %{customdata[1]}<br>' +
                      '<b>Vesicle Size:</b> %{marker.size:.2f}% of 80×80×80 Volume<extra></extra>'
    )
    
    # Save the figure
    fig_umap_by_cluster.write_html(os.path.join(output_dir, "umap_cluster_vesicle_size.html"))
    
    # Create 3D scatter plot with bbox coloring and vesicle sizes
    fig_umap_by_bbox = px.scatter_3d(
        merged_df,
        x=projection_3d[:,0],
        y=projection_3d[:,1],
        z=projection_3d[:,2],
        color='bbox_name',
        size='vesicle_cloud_size',
        hover_data=['size_category', 'cluster_str'],
        title="3D Topological Manifold with Bbox Coloring & Vesicle Size (% of 80×80×80 Volume)",
        opacity=0.7
    )
    fig_umap_by_bbox.update_layout(width=1200, height=800)
    
    # Update hover template to show percentages
    fig_umap_by_bbox.update_traces(
        hovertemplate='<b>Bbox:</b> %{color}<br>' +
                      '<b>Size Category:</b> %{customdata[0]}<br>' +
                      '<b>Cluster:</b> %{customdata[1]}<br>' +
                      '<b>Vesicle Size:</b> %{marker.size:.2f}% of 80×80×80 Volume<extra></extra>'
    )
    
    # Save the figure
    fig_umap_by_bbox.write_html(os.path.join(output_dir, "umap_bbox_vesicle_size.html"))
    
    return merged_df, fig_umap_by_cluster, fig_umap_by_bbox


def analyze_vesicle_sizes_by_cluster(merged_df, output_dir):
    """
    Create comprehensive analysis of vesicle sizes by cluster.
    
    Args:
        merged_df: Merged dataframe with features and vesicle sizes
        output_dir: Directory to save results
        
    Returns:
        str: Path to the generated HTML report
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Box Plot
    merged_df['cluster_str'] = 'Cluster ' + merged_df['cluster'].astype(str)
    fig_box = px.box(
        merged_df,
        x='cluster_str',  # Use string version for categorical coloring
        y='vesicle_cloud_size',
        color='cluster_str',
        points="all",
        hover_data=['bbox_name'],
        title="Vesicle Cloud Size Distribution by Cluster (% of 80×80×80 Volume)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_box.update_layout(
        yaxis_title="Vesicle Cloud Size (% of 80×80×80 Volume)"
    )
    
    # Update hover template for box plot points
    fig_box.update_traces(
        hovertemplate='<b>Cluster:</b> %{x}<br>' +
                      '<b>Vesicle Size:</b> %{y:.2f}% of 80×80×80 Volume<br>' +
                      '<b>Bbox:</b> %{customdata[0]}<extra></extra>',
        selector=dict(type='box')
    )

    # Add mean markers
    means = merged_df.groupby('cluster_str')['vesicle_cloud_size'].mean()
    mean_trace = go.Scatter(
        x=means.index,
        y=means.values,
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name='Mean',
        hovertemplate='<b>Cluster:</b> %{x}<br>' +
                      '<b>Mean Vesicle Size:</b> %{y:.2f}% of 80×80×80 Volume<extra></extra>'
    )
    fig_box.add_trace(mean_trace)

    # Violin Plot
    fig_violin = go.Figure()
    for cluster in sorted(merged_df['cluster'].unique()):
        cluster_str = f'Cluster {cluster}'
        cluster_data = merged_df[merged_df['cluster'] == cluster]
        fig_violin.add_trace(go.Violin(
            x=cluster_data['cluster_str'],
            y=cluster_data['vesicle_cloud_size'],
            name=cluster_str,
            box_visible=True,
            meanline_visible=True,
            hovertemplate='<b>Cluster:</b> %{x}<br>' +
                        '<b>Vesicle Size:</b> %{y:.2f}% of 80×80×80 Volume<extra></extra>'
        ))
    
    fig_violin.update_layout(
        title="Violin Plots of Vesicle Size Distributions per Cluster (% of 80×80×80 Volume)",
        xaxis_title="Cluster",
        yaxis_title="Vesicle Cloud Size (% of 80×80×80 Volume)"
    )

    # Statistical Summary Table
    stats_df = merged_df.groupby('cluster')['vesicle_cloud_size'].agg(
        Mean=np.mean,
        Median=np.median,
        Std=np.std,
        Min=np.min,
        Max=np.max,
        Count='count'
    ).reset_index()

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=stats_df.columns.tolist()),
        cells=dict(values=stats_df.values.T))
    ])
    
    fig_table.update_layout(title="Statistical Summary by Cluster")

    # Enhanced Violin Plot with Distribution Metrics
    fig_enhanced_violin = go.Figure()
    for cluster in sorted(merged_df['cluster'].unique()):
        cluster_str = f'Cluster {cluster}'
        cluster_data = merged_df[merged_df['cluster'] == cluster]

        # Add violin plot
        fig_enhanced_violin.add_trace(go.Violin(
            x=cluster_data['cluster_str'],
            y=cluster_data['vesicle_cloud_size'],
            name=cluster_str,
            box_visible=True,
            meanline_visible=True,
            points="all",
            pointpos=0,
            jitter=0.05
        ))

        # Add statistical annotations
        stats_text = (f"Mean: {cluster_data['vesicle_cloud_size'].mean():.2f}%<br>"
                     f"Median: {cluster_data['vesicle_cloud_size'].median():.2f}%<br>"
                     f"SD: {cluster_data['vesicle_cloud_size'].std():.2f}%")

        fig_enhanced_violin.add_annotation(
            x=cluster_str,
            y=cluster_data['vesicle_cloud_size'].max() * 1.1,
            text=stats_text,
            showarrow=False,
            font=dict(size=9)
        )
    
    fig_enhanced_violin.update_layout(
        title="Enhanced Violin Plots with Distribution Metrics",
        xaxis_title="Cluster",
        yaxis_title="Vesicle Cloud Size (%)"
    )

    # Statistical Comparison Matrix
    clusters = sorted(merged_df['cluster'].unique())
    effect_size_matrix = np.zeros((len(clusters), len(clusters)))

    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i != j:
                data1 = merged_df[merged_df['cluster'] == cluster1]['vesicle_cloud_size']
                data2 = merged_df[merged_df['cluster'] == cluster2]['vesicle_cloud_size']
                effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.std()**2 + data2.std()**2)/2)
                effect_size_matrix[i, j] = effect_size

    # Create heatmap for effect sizes
    fig_effect_size = go.Figure(data=go.Heatmap(
        z=effect_size_matrix,
        x=[f'Cluster {c}' for c in clusters],
        y=[f'Cluster {c}' for c in clusters],
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Cohen's d"),
        hoverongaps=False
    ))

    # Add annotations for effect sizes
    annotations = []
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            annotations.append(
                dict(
                    x=f'Cluster {cluster2}',
                    y=f'Cluster {cluster1}',
                    text=f"{effect_size_matrix[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color='white' if abs(effect_size_matrix[i, j]) > 0.5 else 'black')
                )
            )

    fig_effect_size.update_layout(
        title="Pairwise Effect Size Comparison (Cohen's d)",
        xaxis_title="Cluster",
        yaxis_title="Cluster",
        annotations=annotations,
        width=800,
        height=800
    )

    # Cumulative Distribution Plot
    fig_cdf = go.Figure()
    for cluster in sorted(merged_df['cluster'].unique()):
        cluster_str = f'Cluster {cluster}'
        cluster_data = merged_df[merged_df['cluster'] == cluster]['vesicle_cloud_size']
        hist, bin_edges = np.histogram(cluster_data, bins=50, density=True)
        cdf = np.cumsum(hist * np.diff(bin_edges))

        fig_cdf.add_trace(go.Scatter(
            x=bin_edges[1:],
            y=cdf,
            mode='lines',
            name=cluster_str,
            opacity=0.7
        ))
    
    fig_cdf.update_layout(
        title="Cumulative Distribution Functions by Cluster",
        xaxis_title="Vesicle Cloud Size (% of Subvolume)",
        yaxis_title="Cumulative Probability"
    )

    # Advanced Distribution Comparison (Kernel Density Estimation)
    fig_kde = go.Figure()
    for cluster in clusters:
        cluster_str = f'Cluster {cluster}'
        data = merged_df[merged_df['cluster'] == cluster]['vesicle_cloud_size']
        if len(data) > 1:  # Only create KDE if there are at least 2 data points
            kernel = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 100)
            fig_kde.add_trace(go.Scatter(
                x=x,
                y=kernel(x),
                mode='lines',
                name=cluster_str,
                fill='tozeroy'
            ))
        else:
            # For clusters with only one point, just add a vertical line
            x_value = data.iloc[0] if len(data) > 0 else 0
            fig_kde.add_trace(go.Scatter(
                x=[x_value, x_value],
                y=[0, 1],  # Arbitrary height
                mode='lines',
                name=f'Cluster {cluster} (single point)',
                line=dict(width=2, dash='dash')
            ))
    
    fig_kde.update_layout(
        title="Probability Density Functions by Cluster",
        xaxis_title="Vesicle Cloud Size (% of Subvolume)",
        yaxis_title="Density"
    )

    # Bayesian Hierarchical Modeling
    print("Fitting Bayesian Hierarchical Model...")
    try:
        mixed_model = sm.MixedLM.from_formula(
            'vesicle_cloud_size ~ cluster',
            groups=merged_df['bbox_name'],
            data=merged_df
        ).fit()
        model_summary = mixed_model.summary().as_html()
    except Exception as e:
        print(f"Error fitting mixed model: {e}")
        model_summary = f"<p>Error fitting mixed model: {e}</p>"

    # Update individual cluster plots with detailed statistics
    individual_plots = []
    for cluster in sorted(merged_df['cluster'].unique()):
        cluster_data = merged_df[merged_df['cluster'] == cluster]
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=cluster_data['vesicle_cloud_size'],
            name='Box Plot',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker_color='blue',
            line_color='darkblue'
        ))
        
        # Add mean and std indicators
        mean_val = cluster_data['vesicle_cloud_size'].mean()
        
        # Add annotation with statistics
        stats_text = (f"Mean: {cluster_data['vesicle_cloud_size'].mean():.2f}% of 80×80×80 Volume<br>"
                     f"Median: {cluster_data['vesicle_cloud_size'].median():.2f}% of 80×80×80 Volume<br>"
                     f"SD: {cluster_data['vesicle_cloud_size'].std():.2f}% of 80×80×80 Volume")
        
        fig.add_annotation(
            x=0.95,
            y=cluster_data['vesicle_cloud_size'].max() * 1.1,
            xref="paper",
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="darkblue",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"Vesicle Cloud Size Distribution - Cluster {cluster} (% of 80×80×80 Volume)",
            xaxis_title="Samples",
            yaxis_title="Vesicle Cloud Size (% of 80×80×80 Volume)"
        )
        
        individual_plots.append(fig)

    # Update histograms for each cluster with percentage units
    histograms = []
    for cluster in sorted(merged_df['cluster'].unique()):
        cluster_data = merged_df[merged_df['cluster'] == cluster]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cluster_data['vesicle_cloud_size'],
            nbinsx=10,
            marker_color='blue'
        ))
        
        fig.update_layout(
            title=f"Vesicle Cloud Size Histogram - Cluster {cluster} (% of 80×80×80 Volume)",
            xaxis_title="Vesicle Cloud Size (% of 80×80×80 Volume)",
            yaxis_title="Count"
        )
        
        histograms.append(fig)

    # Create HTML report
    html_filename = os.path.join(output_dir, "vesicle_size_analysis.html")
    with open(html_filename, "w") as f:
        f.write("<html><head><title>Vesicle Analysis Results</title></head><body>")
        f.write("<h1>Vesicle Cloud Size Analysis Results</h1><hr>")
        f.write("<h2>Vesicle Cloud Size Distribution by Cluster (% of 80×80×80 Volume)</h2>")
        f.write(fig_box.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Violin Plots of Vesicle Size Distributions per Cluster (% of 80×80×80 Volume)</h2>")
        f.write(fig_violin.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Statistical Summary by Cluster</h2>")
        f.write(fig_table.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Enhanced Violin Plots with Distribution Metrics</h2>")
        f.write(fig_enhanced_violin.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Pairwise Effect Size Comparison (Cohen's d)</h2>")
        f.write(fig_effect_size.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Cumulative Distribution Functions by Cluster</h2>")
        f.write(fig_cdf.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Individual Cluster Plots</h2>")
        for fig in individual_plots:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Vesicle Cloud Size Histograms by Cluster</h2>")
        for fig in histograms:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Bayesian Hierarchical Modeling Summary</h2>")
        f.write(model_summary)
        f.write("<hr></body></html>")

    print(f"HTML report saved to {html_filename}")
    return html_filename


def count_bboxes_in_clusters(features_df):
    """
    Count the occurrences of each bounding box in each cluster.
    Returns a DataFrame with the counts.
    """
    # Create a pivot table where rows are clusters and columns are bounding boxes
    cluster_counts = features_df.groupby(['cluster', 'bbox_name']).size().unstack(fill_value=0)
    return cluster_counts


def plot_bboxes_in_clusters(cluster_counts, output_dir):
    """
    Plot various visualizations showing the count of each bounding box 
    in each cluster using Plotly.
    
    Args:
        cluster_counts: DataFrame of counts of bounding boxes in each cluster
        output_dir: Directory to save the results
        
    Returns:
        str: Path to the generated HTML report
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar Chart: Number of Bounding Boxes in Each Cluster
    bar_fig = go.Figure()
    for bbox_name in cluster_counts.columns:
        bar_fig.add_trace(go.Bar(
            x=cluster_counts.index,
            y=cluster_counts[bbox_name],
            name=bbox_name,
            hoverinfo='x+y+name',  # Show both x (cluster) and y (count)
        ))

    bar_fig.update_layout(
        barmode='stack',
        title='Number of Bounding Boxes in Each Cluster',
        xaxis_title='Cluster',
        yaxis_title='Count of Bounding Boxes',
        showlegend=True
    )

    # Box Plot: Distribution of Bounding Boxes per Cluster
    box_df = cluster_counts.melt(var_name="Bounding Box", value_name="Count")
    box_fig = px.box(
        box_df,
        x="Bounding Box",
        y="Count",
        color="Bounding Box",
        title="Distribution of Bounding Boxes per Cluster"
    )

    # Scatter Plot: Show Relationship between Bounding Box and Cluster
    # Melt the data into a long format for the scatter plot
    scatter_df = cluster_counts.reset_index().melt(id_vars=["cluster"], var_name="Bounding Box", value_name="Count")
    scatter_fig = px.scatter(
        scatter_df,
        x="cluster",
        y="Count",
        color="Bounding Box",  # Color by Bounding Box
        labels={"cluster": "Cluster", "Count": "Bounding Box Count", "color": "Bounding Box"},
        title="Scatter Plot: Bounding Box Count vs Cluster"
    )

    # Pie Chart: Proportion of Bounding Boxes in Each Cluster
    pie_data = cluster_counts.sum(axis=0).reset_index()
    pie_data.columns = ["bbox_name", "Count"]
    pie_fig = px.pie(
        pie_data,
        names="bbox_name",
        values="Count",
        title="Proportion of Bounding Boxes in Each Cluster"
    )

    # Create HTML report
    html_filename = os.path.join(output_dir, "bbox_cluster_analysis.html")
    with open(html_filename, 'w') as f:
        f.write("<html><body>")
        f.write("<h1>Bounding Box Distribution in Clusters Analysis</h1><hr>")
        f.write("<h2>Number of Bounding Boxes in Each Cluster</h2>")
        f.write(bar_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Distribution of Bounding Boxes per Cluster</h2>")
        f.write(box_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Scatter Plot: Bounding Box Count vs Cluster</h2>")
        f.write(scatter_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<h2>Proportion of Bounding Boxes in Each Cluster</h2>")
        f.write(pie_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<hr></body></html>")

    print(f"HTML report saved to {html_filename}")
    return html_filename 