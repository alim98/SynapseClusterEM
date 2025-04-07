import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import umap
import seaborn as sns
from tqdm import tqdm

# Import from main config
from synapse import config

def load_data():
    """
    Load the manual annotations and VGG features
    """
    print("Loading data...")
    
    # Set up paths
    results_dir = 'manual/clustering_results_final'
    
    # Load manual clustering results
    manual_file = os.path.join(results_dir, "manual_clustered_samples.csv")
    if not os.path.exists(manual_file):
        raise FileNotFoundError(f"Manual clustering file not found: {manual_file}")
    manual_df = pd.read_csv(manual_file)
    
    # Load VGG features
    vgg_file = os.path.join(results_dir, "vgg_features.csv")
    if not os.path.exists(vgg_file):
        raise FileNotFoundError(f"VGG features file not found: {vgg_file}")
    vgg_df = pd.read_csv(vgg_file)
    
    # Merge DataFrames
    merged_df = manual_df[['bbox_name', 'Var1', 'Manual_Cluster']].merge(
        vgg_df, on=['bbox_name', 'Var1']
    )
    
    print(f"Loaded {len(merged_df)} samples with manual clusters and VGG features")
    return merged_df

def evaluate_feature_subset(features, labels, feature_names=None, method='umap', n_clusters=2):
    """
    Evaluate how well a subset of features creates clusters that match manual annotations
    
    Args:
        features: Feature matrix
        labels: Ground truth manual cluster labels
        feature_names: List of feature names (for reporting)
        method: 'umap' or 'pca' for dimensionality reduction
        n_clusters: Number of clusters to create
        
    Returns:
        scores: Dictionary of evaluation metrics
        projections: Projected coordinates
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        projection = reducer.fit_transform(features_scaled)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        projection = reducer.fit_transform(features_scaled)
    
    # Cluster the projection
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_clusters = kmeans.fit_predict(projection)
    
    # Calculate evaluation metrics
    ari = adjusted_rand_score(labels, predicted_clusters)
    nmi = normalized_mutual_info_score(labels, predicted_clusters)
    v_measure = v_measure_score(labels, predicted_clusters)
    
    # Calculate average score
    avg_score = (ari + nmi + v_measure) / 3
    
    return {
        'ari': ari,
        'nmi': nmi,
        'v_measure': v_measure,
        'avg_score': avg_score,
        'features': feature_names,
        'n_features': features.shape[1]
    }, projection, predicted_clusters

def find_best_features_mutual_info(df, output_dir, projection_method='umap'):
    """
    Find the best features using mutual information
    """
    print(f"Finding best features using mutual information and {projection_method}...")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if 'feat_' in col]
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df['Manual_Cluster'].values
    
    # Calculate mutual information between each feature and manual clusters
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create a DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'Feature': feature_cols,
        'MI_Score': mi_scores
    })
    
    # Sort by mutual information score
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    
    # Save MI scores
    mi_df.to_csv(os.path.join(output_dir, f"mutual_info_scores_{projection_method}.csv"), index=False)
    
    # Plot top 20 features by MI score
    plt.figure(figsize=(12, 8))
    sns.barplot(data=mi_df.head(20), x='MI_Score', y='Feature')
    plt.title(f"Top 20 Features by Mutual Information with Manual Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mutual_info_top_features_{projection_method}.png"))
    
    # Try different numbers of top features
    k_values = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    k_values = [k for k in k_values if k <= len(feature_cols)]
    
    results = []
    best_score = -1
    best_k = 0
    best_projection = None
    best_clusters = None
    
    for k in tqdm(k_values, desc=f"Testing feature subsets using {projection_method}"):
        # Select top k features
        top_features = mi_df['Feature'].head(k).tolist()
        X_subset = df[top_features].values
        
        # Evaluate the feature subset
        score, projection, clusters = evaluate_feature_subset(
            X_subset, y, top_features, method=projection_method, n_clusters=len(np.unique(y))
        )
        
        # Add number of features to results
        score['k'] = k
        results.append(score)
        
        # Track best score
        if score['avg_score'] > best_score:
            best_score = score['avg_score']
            best_k = k
            best_projection = projection
            best_clusters = clusters
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, f"feature_selection_results_{projection_method}.csv"), index=False)
    
    # Plot scores vs number of features
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['ari'], label='ARI')
    plt.plot(results_df['k'], results_df['nmi'], label='NMI')
    plt.plot(results_df['k'], results_df['v_measure'], label='V-measure')
    plt.plot(results_df['k'], results_df['avg_score'], label='Average', linewidth=2)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('Number of features')
    plt.ylabel('Score')
    plt.title(f'Feature Selection Performance ({projection_method})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"feature_selection_scores_{projection_method}.png"))
    
    print(f"Best number of features: {best_k} with score: {best_score:.4f}")
    
    # Select best features
    best_features = mi_df['Feature'].head(best_k).tolist()
    
    # Project the data with best features and visualize
    visualize_best_features(df, best_features, best_projection, best_clusters, output_dir, projection_method)
    
    return best_features, best_score

def visualize_best_features(df, best_features, projection, clusters, output_dir, method):
    """
    Visualize the projection with the best features
    """
    print(f"Visualizing projection with {len(best_features)} best features...")
    
    # Create a DataFrame with projection coordinates
    proj_df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'Manual_Cluster': df['Manual_Cluster'].values,
        'Predicted_Cluster': clusters,
        'bbox_name': df['bbox_name'].values,
        'Var1': df['Var1'].values
    })
    
    # Save the projection
    proj_df.to_csv(os.path.join(output_dir, f"best_feature_projection_{method}.csv"), index=False)
    
    # Create colors for manual clusters
    unique_manual = sorted(proj_df['Manual_Cluster'].unique())
    manual_colors = {c: plt.cm.viridis(i/max(1, len(unique_manual)-1)) for i, c in enumerate(unique_manual)}
    
    # Create colors for predicted clusters
    unique_predicted = sorted(proj_df['Predicted_Cluster'].unique())
    predicted_colors = {c: plt.cm.plasma(i/max(1, len(unique_predicted)-1)) for i, c in enumerate(unique_predicted)}
    
    # Plot manual clusters
    plt.figure(figsize=(12, 10))
    for cluster, group in proj_df.groupby('Manual_Cluster'):
        plt.scatter(
            group['x'],
            group['y'],
            label=f"Manual Cluster {cluster}",
            color=manual_colors[cluster],
            s=100,
            alpha=0.8
        )
    
    # Add synapse labels
    for i, row in proj_df.iterrows():
        label = row['Var1'].split('_')[-1]  # Just use the last part of the name
        plt.annotate(
            label,
            (row['x'], row['y']),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title(f"{method.upper()} Projection of {len(best_features)} Best Features (Manual Clusters)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"best_features_{method}_manual_clusters.png"))
    
    # Plot predicted clusters
    plt.figure(figsize=(12, 10))
    for cluster, group in proj_df.groupby('Predicted_Cluster'):
        plt.scatter(
            group['x'],
            group['y'],
            label=f"Predicted Cluster {cluster}",
            color=predicted_colors[cluster],
            s=100,
            alpha=0.8
        )
    
    # Add synapse labels
    for i, row in proj_df.iterrows():
        label = row['Var1'].split('_')[-1]
        plt.annotate(
            label,
            (row['x'], row['y']),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title(f"{method.upper()} Projection of {len(best_features)} Best Features (Predicted Clusters)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"best_features_{method}_predicted_clusters.png"))
    
    # Plot confusion between manual and predicted
    plt.figure(figsize=(10, 8))
    conf_matrix = pd.crosstab(
        proj_df['Manual_Cluster'], 
        proj_df['Predicted_Cluster'],
        rownames=['Manual'],
        colnames=['Predicted']
    )
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title(f"Confusion Matrix: Manual vs Predicted Clusters\n{len(best_features)} Best Features, {method.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"best_features_{method}_confusion.png"))
    
    # Save the best features
    with open(os.path.join(output_dir, f"best_features_{method}.txt"), 'w') as f:
        f.write("Best Features:\n")
        for i, feature in enumerate(best_features):
            f.write(f"{i+1}. {feature}\n")

def find_best_features_genetic(df, output_dir, projection_method='umap', population_size=20, generations=30):
    """
    Find the best features using a genetic algorithm
    """
    print(f"Finding best features using genetic algorithm and {projection_method}...")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if 'feat_' in col]
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df['Manual_Cluster'].values
    
    # For simplicity, we'll use this as a starting point:
    # First use mutual information to get the top 100 features
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create a DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'Feature': feature_cols,
        'MI_Score': mi_scores
    })
    
    # Sort by mutual information score
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    
    # Take top 100 features (or fewer if there aren't that many)
    top_k = min(100, len(feature_cols))
    candidate_features = mi_df['Feature'].head(top_k).tolist()
    
    # Set up genetic algorithm
    from deap import base, creator, tools, algorithms
    import random
    
    # Create fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(candidate_features))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evalFeatureSubset(individual):
        # Get selected features
        selected_features = [candidate_features[i] for i in range(len(individual)) if individual[i] == 1]
        
        # If no features are selected, return a very low score
        if len(selected_features) == 0:
            return 0.0,
        
        # Extract the selected features
        X_subset = df[selected_features].values
        
        # Evaluate the feature subset
        score, _, _ = evaluate_feature_subset(
            X_subset, y, selected_features, method=projection_method, n_clusters=len(np.unique(y))
        )
        
        return score['avg_score'],
    
    # Operator registration
    toolbox.register("evaluate", evalFeatureSubset)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Keep track of the best individual
    hof = tools.HallOfFame(1)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the algorithm
    print(f"Running genetic algorithm for {generations} generations...")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    # Get the best individual
    best_individual = hof[0]
    
    # Get the selected features
    best_features = [candidate_features[i] for i in range(len(best_individual)) if best_individual[i] == 1]
    
    print(f"Best features selected: {len(best_features)}")
    
    # Evaluate the best feature subset
    X_subset = df[best_features].values
    best_score, projection, clusters = evaluate_feature_subset(
        X_subset, y, best_features, method=projection_method, n_clusters=len(np.unique(y))
    )
    
    # Save the best features
    with open(os.path.join(output_dir, f"genetic_best_features_{projection_method}.txt"), 'w') as f:
        f.write(f"Best {len(best_features)} Features (Genetic Algorithm):\n")
        for i, feature in enumerate(best_features):
            f.write(f"{i+1}. {feature}\n")
    
    # Plot fitness over generations
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_avgs = log.select("avg")
    fit_maxs = log.select("max")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    plt.plot(gen, fit_avgs, "r-", label="Average Fitness")
    plt.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Higher is Better)")
    plt.legend(loc="best")
    plt.title(f"Genetic Algorithm: Fitness Evolution ({projection_method})")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"genetic_algorithm_fitness_{projection_method}.png"))
    
    # Visualize the best features
    visualize_best_features(df, best_features, projection, clusters, output_dir, f"genetic_{projection_method}")
    
    return best_features, best_score['avg_score']

def compare_feature_methods(df, output_dir):
    """
    Compare different feature selection methods using bar charts
    """
    print("Comparing feature selection methods...")
    
    # Results structure
    results = []
    
    # Run all methods
    # Mutual Information with UMAP
    mi_umap_features, mi_umap_score = find_best_features_mutual_info(df, output_dir, 'umap')
    results.append({
        'Method': 'MI+UMAP',
        'Score': mi_umap_score,
        'Features': len(mi_umap_features)
    })
    
    # Mutual Information with PCA
    mi_pca_features, mi_pca_score = find_best_features_mutual_info(df, output_dir, 'pca')
    results.append({
        'Method': 'MI+PCA',
        'Score': mi_pca_score,
        'Features': len(mi_pca_features)
    })
    
    # Genetic Algorithm with UMAP
    genetic_umap_features, genetic_umap_score = find_best_features_genetic(df, output_dir, 'umap')
    results.append({
        'Method': 'Genetic+UMAP',
        'Score': genetic_umap_score,
        'Features': len(genetic_umap_features)
    })
    
    # Genetic Algorithm with PCA
    genetic_pca_features, genetic_pca_score = find_best_features_genetic(df, output_dir, 'pca')
    results.append({
        'Method': 'Genetic+PCA',
        'Score': genetic_pca_score,
        'Features': len(genetic_pca_features)
    })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save comparison results
    results_df.to_csv(os.path.join(output_dir, "feature_selection_comparison.csv"), index=False)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=results_df, x='Method', y='Score')
    
    # Add the number of features as text on the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{results_df.iloc[i]['Features']} features", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', 
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.title("Comparison of Feature Selection Methods")
    plt.ylabel("Score (Higher is Better)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_selection_comparison.png"))
    
    print("Feature selection comparison complete")

def main():
    # Set up directory for output
    output_dir = 'manual/feature_selection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    merged_df = load_data()
    
    # For quick testing, uncomment one of these:
    # find_best_features_mutual_info(merged_df, output_dir, 'umap')
    # find_best_features_mutual_info(merged_df, output_dir, 'pca')
    # find_best_features_genetic(merged_df, output_dir, 'umap')
    # find_best_features_genetic(merged_df, output_dir, 'pca')
    
    # Run all methods and compare
    compare_feature_methods(merged_df, output_dir)
    
    print(f"Feature selection complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 