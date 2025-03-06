import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns

def compare_intra_inter_presynapse_distances(features_df, presynapse_groups, output_dir):
    """
    Compare distances between synapses sharing the same presynapse ID versus 
    distances between synapses with different presynapse IDs.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing feature data
        presynapse_groups (dict): Dictionary with presynapse IDs as keys and lists of synapse indices as values
        output_dir (str): Directory to save visualizations
    
    Returns:
        dict: Dictionary with summary statistics of the comparison
    """
    print("Comparing intra-presynapse and inter-presynapse distances")
    
    # Extract feature columns
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    if not feature_cols:
        print("No feature columns found, cannot compute distances")
        return None
    
    # Get features matrix
    features = features_df[feature_cols].values
    
    # Create a mapping from index to presynapse ID
    index_to_presynapse = {}
    for pre_id, indices in presynapse_groups.items():
        for idx in indices:
            index_to_presynapse[idx] = pre_id
    
    # Lists to store distances
    intra_presynapse_distances = []
    inter_presynapse_distances = []
    
    # Calculate pairwise distances
    print("Calculating pairwise distances between all synapses...")
    all_distances = euclidean_distances(features)
    
    # For each pair of synapses, determine if they share the same presynapse ID
    for i in range(len(features_df)):
        for j in range(i+1, len(features_df)):
            # Skip if either synapse is not part of any presynapse group
            if i not in index_to_presynapse or j not in index_to_presynapse:
                continue
            
            distance = all_distances[i, j]
            
            # Check if they share the same presynapse ID
            if index_to_presynapse[i] == index_to_presynapse[j]:
                intra_presynapse_distances.append(distance)
            else:
                inter_presynapse_distances.append(distance)
    
    # Calculate summary statistics
    if intra_presynapse_distances and inter_presynapse_distances:
        intra_mean = np.mean(intra_presynapse_distances)
        intra_std = np.std(intra_presynapse_distances)
        intra_min = np.min(intra_presynapse_distances)
        intra_max = np.max(intra_presynapse_distances)
        
        inter_mean = np.mean(inter_presynapse_distances)
        inter_std = np.std(inter_presynapse_distances)
        inter_min = np.min(inter_presynapse_distances)
        inter_max = np.max(inter_presynapse_distances)
        
        ratio = intra_mean / inter_mean if inter_mean > 0 else 0
        
        print(f"Intra-presynapse distance (mean ± std): {intra_mean:.4f} ± {intra_std:.4f}")
        print(f"Inter-presynapse distance (mean ± std): {inter_mean:.4f} ± {inter_std:.4f}")
        print(f"Ratio (intra/inter): {ratio:.4f}")
        
        # Create a histogram to compare distributions
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        bins = np.linspace(min(intra_min, inter_min), max(intra_max, inter_max), 30)
        plt.hist(intra_presynapse_distances, bins=bins, alpha=0.7, label=f'Same Presynapse (n={len(intra_presynapse_distances)})')
        plt.hist(inter_presynapse_distances, bins=bins, alpha=0.7, label=f'Different Presynapse (n={len(inter_presynapse_distances)})')
        
        # Add vertical lines for means
        plt.axvline(intra_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Same Presynapse Mean: {intra_mean:.4f}')
        plt.axvline(inter_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Different Presynapse Mean: {inter_mean:.4f}')
        
        plt.title('Distribution of Distances: Same vs. Different Presynapse ID')
        plt.xlabel('Euclidean Distance in Feature Space')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        # Save histogram
        hist_path = os.path.join(output_dir, "distance_comparison_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        
        # Create box plot to compare distributions
        plt.figure(figsize=(10, 8))
        
        # Convert to DataFrame for easier plotting
        distance_df = pd.DataFrame({
            'Distance': intra_presynapse_distances + inter_presynapse_distances,
            'Group': ['Same Presynapse'] * len(intra_presynapse_distances) + 
                     ['Different Presynapse'] * len(inter_presynapse_distances)
        })
        
        # Create box plot with individual points
        sns.boxplot(x='Group', y='Distance', data=distance_df)
        sns.stripplot(x='Group', y='Distance', data=distance_df, 
                     size=4, color='.3', alpha=0.3)
        
        plt.title('Distance Distribution: Same vs. Different Presynapse ID')
        plt.ylabel('Euclidean Distance in Feature Space')
        plt.tight_layout()
        
        # Save box plot
        box_path = os.path.join(output_dir, "distance_comparison_boxplot.png")
        plt.savefig(box_path, dpi=300)
        plt.close()
        
        # Create violin plot for more detailed distribution view
        plt.figure(figsize=(10, 8))
        sns.violinplot(x='Group', y='Distance', data=distance_df, inner='quartile')
        plt.title('Distance Distribution: Same vs. Different Presynapse ID')
        plt.ylabel('Euclidean Distance in Feature Space')
        plt.tight_layout()
        
        # Save violin plot
        violin_path = os.path.join(output_dir, "distance_comparison_violinplot.png")
        plt.savefig(violin_path, dpi=300)
        plt.close()
        
        # Generate statistics per presynapse ID
        per_presynapse_stats = {}
        for pre_id, indices in presynapse_groups.items():
            if len(indices) > 1:  # Need at least 2 synapses to calculate distances
                # Calculate intra-presynapse distances for this ID
                distances = []
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        distances.append(all_distances[idx1, idx2])
                
                # Calculate average distance to other presynapses
                other_distances = []
                for idx1 in indices:
                    for pre_id2, indices2 in presynapse_groups.items():
                        if pre_id != pre_id2:
                            for idx2 in indices2:
                                other_distances.append(all_distances[idx1, idx2])
                
                # Store statistics
                per_presynapse_stats[pre_id] = {
                    'intra_mean': np.mean(distances) if distances else 0,
                    'intra_std': np.std(distances) if distances else 0,
                    'inter_mean': np.mean(other_distances) if other_distances else 0,
                    'inter_std': np.std(other_distances) if other_distances else 0,
                    'ratio': np.mean(distances) / np.mean(other_distances) if other_distances and np.mean(other_distances) > 0 else 0,
                    'num_synapses': len(indices)
                }
        
        # Create a bar plot comparing the intra/inter ratio for each presynapse ID
        plt.figure(figsize=(14, 10))
        pre_ids = list(per_presynapse_stats.keys())
        ratios = [per_presynapse_stats[pre_id]['ratio'] for pre_id in pre_ids]
        
        # Sort by ratio
        sorted_indices = np.argsort(ratios)
        sorted_pre_ids = [pre_ids[i] for i in sorted_indices]
        sorted_ratios = [ratios[i] for i in sorted_indices]
        
        # Color bars by ratio (lower is better)
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_ratios)))
        
        # Create bar plot
        bars = plt.bar(range(len(sorted_pre_ids)), sorted_ratios, color=colors)
        
        # Add a line for the overall average ratio
        plt.axhline(ratio, color='red', linestyle='--', label=f'Overall Average: {ratio:.4f}')
        
        # Add number of synapses as text on each bar
        for i, (pre_id, bar) in enumerate(zip(sorted_pre_ids, bars)):
            num_syn = per_presynapse_stats[pre_id]['num_synapses']
            plt.text(i, 0.02, f"n={num_syn}", ha='center', va='bottom', color='black')
        
        plt.title('Distance Ratio (Intra/Inter) by Presynapse ID\nLower is Better')
        plt.xlabel('Presynapse ID')
        plt.ylabel('Ratio of Intra-Presynapse Distance to Inter-Presynapse Distance')
        plt.xticks(range(len(sorted_pre_ids)), sorted_pre_ids, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # Save bar plot
        bar_path = os.path.join(output_dir, "distance_ratio_by_presynapse.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
        
        # Create a scatter plot of intra vs inter distances
        plt.figure(figsize=(10, 8))
        intra_means = [per_presynapse_stats[pre_id]['intra_mean'] for pre_id in pre_ids]
        inter_means = [per_presynapse_stats[pre_id]['inter_mean'] for pre_id in pre_ids]
        sizes = [per_presynapse_stats[pre_id]['num_synapses'] * 20 for pre_id in pre_ids]  # Scale by number of synapses
        
        # Draw diagonal line (y=x)
        min_val = min(min(intra_means), min(inter_means))
        max_val = max(max(intra_means), max(inter_means))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Create scatter plot
        scatter = plt.scatter(intra_means, inter_means, c=ratios, cmap='viridis', 
                            s=sizes, alpha=0.7, edgecolors='black')
        
        # Add presynapse IDs as labels
        for i, pre_id in enumerate(pre_ids):
            plt.annotate(str(pre_id), (intra_means[i], inter_means[i]), 
                        fontsize=8, ha='center', va='center')
        
        plt.colorbar(scatter, label='Intra/Inter Ratio (lower is better)')
        plt.title('Intra-Presynapse vs. Inter-Presynapse Distances')
        plt.xlabel('Average Distance Between Synapses with Same Presynapse ID')
        plt.ylabel('Average Distance to Synapses with Different Presynapse ID')
        plt.tight_layout()
        
        # Save scatter plot
        scatter_path = os.path.join(output_dir, "intra_vs_inter_distance_scatter.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        
        # Return summary
        summary = {
            'intra_mean': intra_mean,
            'intra_std': intra_std,
            'inter_mean': inter_mean,
            'inter_std': inter_std,
            'ratio': ratio,
            'n_intra': len(intra_presynapse_distances),
            'n_inter': len(inter_presynapse_distances),
            'per_presynapse': per_presynapse_stats,
            'plots': {
                'histogram': hist_path,
                'boxplot': box_path,
                'violinplot': violin_path,
                'barplot': bar_path,
                'scatterplot': scatter_path
            }
        }
        
        return summary
    else:
        print("Not enough data to compare distances")
        return None

def add_distance_comparison_to_report(report_path, distance_comparison, output_dir):
    """
    Add distance comparison results to an existing HTML report.
    
    Args:
        report_path (str): Path to the existing HTML report
        distance_comparison (dict): Results from distance comparison analysis
        output_dir (str): Directory where the report is saved
    """
    if not os.path.exists(report_path) or not distance_comparison:
        print("Report path does not exist or no distance comparison data provided")
        return
    
    print(f"Adding distance comparison to report at {report_path}")
    
    # Read the existing report
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find the position to insert the new content (after the summary div)
    insert_position = content.find('</div>', content.find('<div class="summary">')) + 6
    
    # Prepare the content to insert
    insert_content = f"""
    <div class="visualization">
        <h2>Distance Comparison Analysis</h2>
        <div class="summary">
            <h3>Summary</h3>
            <p>Average distance between synapses with the same presynapse ID: {distance_comparison['intra_mean']:.4f} ± {distance_comparison['intra_std']:.4f}</p>
            <p>Average distance between synapses with different presynapse IDs: {distance_comparison['inter_mean']:.4f} ± {distance_comparison['inter_std']:.4f}</p>
            <p>Ratio (intra/inter): {distance_comparison['ratio']:.4f} (lower is better - indicates synapses with same presynapse are more similar)</p>
            <p>Number of intra-presynapse comparisons: {distance_comparison['n_intra']}</p>
            <p>Number of inter-presynapse comparisons: {distance_comparison['n_inter']}</p>
        </div>
    """
    
    # Add visualizations if available
    if 'plots' in distance_comparison:
        plots = distance_comparison['plots']
        
        # Add histogram
        if 'histogram' in plots:
            hist_path = os.path.relpath(plots['histogram'], output_dir)
            insert_content += f"""
            <h3>Distance Distribution Comparison</h3>
            <img src="{hist_path}" alt="Distance Distribution Histogram">
            <p>This histogram compares the distribution of distances between synapses sharing the same presynapse ID versus distances between synapses with different presynapse IDs.</p>
            """
        
        # Add boxplot
        if 'boxplot' in plots:
            box_path = os.path.relpath(plots['boxplot'], output_dir)
            insert_content += f"""
            <h3>Distance Box Plot</h3>
            <img src="{box_path}" alt="Distance Box Plot">
            <p>This box plot shows the distribution of distances with individual points overlaid.</p>
            """
        
        # Add violin plot
        if 'violinplot' in plots:
            violin_path = os.path.relpath(plots['violinplot'], output_dir)
            insert_content += f"""
            <h3>Distance Violin Plot</h3>
            <img src="{violin_path}" alt="Distance Violin Plot">
            <p>This violin plot shows the full distribution of distances.</p>
            """
        
        # Add bar plot
        if 'barplot' in plots:
            bar_path = os.path.relpath(plots['barplot'], output_dir)
            insert_content += f"""
            <h3>Distance Ratio by Presynapse ID</h3>
            <img src="{bar_path}" alt="Distance Ratio by Presynapse ID">
            <p>This plot shows the ratio of intra-presynapse to inter-presynapse distance for each presynapse ID. Lower values indicate that synapses sharing this presynapse ID are more similar to each other than to other synapses.</p>
            """
        
        # Add scatter plot
        if 'scatterplot' in plots:
            scatter_path = os.path.relpath(plots['scatterplot'], output_dir)
            insert_content += f"""
            <h3>Intra vs. Inter Presynapse Distances</h3>
            <img src="{scatter_path}" alt="Intra vs. Inter Presynapse Distances">
            <p>This scatter plot compares the average distance between synapses with the same presynapse ID (x-axis) to the average distance to synapses with different presynapse IDs (y-axis). Points below the diagonal line indicate presynapse IDs where synapses are more similar to each other than to other synapses.</p>
            """
    
    insert_content += "</div>"
    
    # Insert the new content
    new_content = content[:insert_position] + insert_content + content[insert_position:]
    
    # Write the updated report
    with open(report_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated report with distance comparison analysis") 